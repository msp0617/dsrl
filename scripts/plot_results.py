"""Learning curves, dip metrics and diagnostics from the run CSVs.

    python scripts/plot_results.py --logs $PROJ/logs --out $PROJ/figures

Runs are grouped by name: can_<group>_s<seed>. Every group with at least one
seed gets a mean +- standard-error success curve; groups are drawn together
on one figure per "axis" (--axes, default: the critic and mix axes).

Time axis. Every curve and metric is on ONLINE environment steps: the
initial evaluation (a random pi_W, taken before any update) sits at online
step 0 and every later evaluation at env_steps - rollout, where rollout is
the initial data-collection phase during which the actor is frozen (Can
24,016, Square 32,016). The raw CSVs keep env_steps; the shift happens in
load_eval. The early window is the first --until_env environment steps
(default 100k), i.e. online 0-75,984 on Can and 0-67,984 on Square.

Outputs in --out:
  success_<axis>.png        success rate vs online steps, mean +- SE over seeds,
                            markers on the evaluations actually run
  success_<axis>_early.png  the same cut to the early window (seed count constant)
  diagnostics_<axis>.png    ent_coef, mu_absmean, w_absmean, w_frac_sat,
                            offline_p, qw_mean vs online steps (train_log.csv)
  qgap_<axis>.png           q_start - mc_return from eval_log.csv, where present
  metrics.csv               per run and per group (mean, SE): step0, min over the
                            early window, dip depth, recovery step, early AUC,
                            value at env 129k, final, t50/t80 (online steps to
                            first reach 0.5 / 0.8, --levels); regret against
                            pi_dp when base_policy_eval.csv exists.
                            min_mean_curve is the minimum of the seed-mean
                            curve read on the common 5k grid (online 5,008 x k)
  summary.csv               one row per group in the time-to-level / AUC / final
                            form, mean +- SE over seeds

Early AUC: per seed, trapezoid over online 0..until of the evaluation curve
with the initial evaluation at 0 (a run whose last evaluation falls short of
the cut-off holds its last value to the cut-off), divided by the window
length; then mean +- SE over seeds.

Only the standard library plus numpy, pandas and matplotlib are needed, so it
runs on Colab as is and on a laptop with the CSVs copied over.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

RUN_RE = re.compile(r"^(?P<task>can|square|lift|transport)_(?P<group>.+)_s(?P<seed>\d+)$")
# Env steps of the initial rollout, during which pi_W is not updated
# (init_rollout_steps x n_envs x act_steps in cfg/robomimic/dsrl_<task>.yaml).
ROLLOUT_ENV = {"can": 24016, "square": 32016}
# Groups keep their Can names; other tasks are prefixed (square_baseline), and
# each axis is drawn on its own so tasks never share a y axis.
DEFAULT_AXES = {
    "critic": ["baseline", "iql", "warmupc", "warmup"],
    "mix": ["baseline", "mix_prefill", "mix_fixed", "mix_linear", "iql_prefill"],
    "square": ["square_baseline", "square_iql", "square_mix_prefill", "square_fixalpha_03", "square_tent12",
               "square_tent6"],
    "alpha": ["baseline", "alr_half", "alr_double", "fixalpha"],
    "sweep": ["baseline", "fixalpha", "fixalpha_003", "fixalpha_01", "fixalpha_03", "fixalpha_1"],
    "adaptive": ["baseline", "iql", "mix_prefill", "fixalpha_03", "tent12", "tent12i", "tent6"],
    "scale": ["baseline", "rs_025", "rs_05", "rs_2", "hardq"],
    "gate": ["baseline", "hardq", "gate_sig", "gate_clk"],
}
DIAG_COLS = ["ent_coef", "mu_absmean", "w_absmean", "w_frac_sat", "log_std_mean", "offline_p", "qw_mean",
             "ratio_ge_gq", "gq_norm"]
GRID_ONLINE = 5008  # spacing of the 5k evaluation schedule in online steps
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


# ------------------------------------------------------------------ loading

def find_runs(logs_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(logs_dir, "*_s[0-9]*"))):
        if not os.path.isdir(path):
            continue
        m = RUN_RE.match(os.path.basename(path))
        if not m or not os.path.exists(os.path.join(path, "eval_log.csv")):
            continue
        group = m.group("group") if m.group("task") == "can" else "%s_%s" % (m.group("task"), m.group("group"))
        runs.setdefault(group, {})[int(m.group("seed"))] = path
    return runs


def group_task(group):
    for task in ("square", "lift", "transport"):
        if group.startswith(task + "_"):
            return task
    return "can"


def run_task(path):
    m = RUN_RE.match(os.path.basename(path))
    return m.group("task") if m else "can"


def rollout_env(task):
    if task not in ROLLOUT_ENV:
        raise SystemExit("no rollout length known for task %r; add it to ROLLOUT_ENV" % task)
    return ROLLOUT_ENV[task]


def to_online(env_steps, task):
    """Online steps: 0 for the initial evaluation, env_steps - rollout after."""
    env_steps = np.asarray(env_steps, float)
    return np.where(env_steps <= 0, 0.0, env_steps - rollout_env(task))


def load_eval(path):
    df = pd.read_csv(os.path.join(path, "eval_log.csv"))
    if "deterministic" in df:
        df = df[df.deterministic == 0]
    df = df.drop_duplicates("env_steps", keep="last").sort_values("env_steps").reset_index(drop=True)
    df["online_steps"] = to_online(df.env_steps, run_task(path))
    return df


def load_train(path):
    f = os.path.join(path, "train_log.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f).sort_values("env_steps").reset_index(drop=True)
    df["online_steps"] = df.env_steps - rollout_env(run_task(path))
    return df


def base_reference(logs_dir, task="can"):
    """Mean and SE of pi_dp under N(0, I) noise, from eval_base_policy.py."""
    name = "base_policy_eval.csv" if task == "can" else "base_policy_eval_%s.csv" % task
    f = os.path.join(logs_dir, name)
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    if len(df) == 0:
        return None
    return float(df.success_rate.mean()), float(df.success_rate.std(ddof=0) / max(np.sqrt(len(df)), 1))


# ------------------------------------------------------------------ metrics

def early_auc(x, y, until):
    """Normalised area of the curve over online 0..until, last value held."""
    keep = x <= until
    xs, ys = x[keep], y[keep]
    if len(xs) < 2:
        return np.nan
    if xs[-1] < until:
        xs, ys = np.append(xs, until), np.append(ys, ys[-1])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(ys, xs) / (xs[-1] - xs[0]))


def run_metrics(ev, until, reference=None, final_points=3, levels=(0.5, 0.8)):
    """Dip statistics of one run's evaluation curve, on online steps.

    t50, t80 (one per entry of `levels`): the first evaluation, online steps,
    at which success reaches the level, the initial evaluation included and
    NaN when the run never gets there. A Can run whose random-actor step-0
    policy already scores above 0.5 has t50 = 0, so on Can t80 is the one
    that carries information.
    """
    x, y = ev.online_steps.to_numpy(float), ev.success_rate.to_numpy(float)
    task = run_task(ev.attrs["path"]) if "path" in ev.attrs else "can"
    step0 = y[0] if len(y) and x[0] == 0 else np.nan
    out = {"step0": step0, "n_evals": int(len(y)),
           "last_online_steps": int(x[-1]) if len(x) else 0,
           "last_env_steps": int(ev.env_steps.iloc[-1]) if len(x) else 0}
    win = (x > 0) & (x <= until)
    if win.any():
        i_min = np.argmin(np.where(win, y, np.inf))
        out["min_in_window"] = float(y[i_min])
        out["min_at"] = int(x[i_min])
        out["dip_depth"] = float(step0 - y[i_min]) if np.isfinite(step0) else np.nan
        # first evaluation after the minimum back at the step-0 level / at pi_dp's level
        after = (x > x[i_min]) & (y >= step0)
        out["recovery_at"] = int(x[np.argmax(after)]) if after.any() and np.isfinite(step0) else np.nan
        if reference is not None:
            after_ref = (x > x[i_min]) & (y >= reference[0])
            out["recovery_pi_dp_at"] = int(x[np.argmax(after_ref)]) if after_ref.any() else np.nan
        out["auc_early"] = early_auc(x, y, until)
    out["final"] = float(np.mean(y[-final_points:])) if len(y) else np.nan
    for level in levels:
        hit = np.nonzero(y >= level)[0]
        out["t%d" % int(round(level * 100))] = int(x[hit[0]]) if len(hit) else np.nan
    # The last evaluation every budget reaches (150k runs stop before the
    # 154k evaluation), so groups with different budgets stay comparable.
    at = 129152 - rollout_env(task)
    if len(x) > 1 and x[-1] >= at - 200:
        out["at_env129k"] = float(np.interp(at, x, y))
    if "mc_return" in ev and ev.mc_return.notna().any():
        out["mc_return_first"] = float(ev.mc_return.dropna().iloc[0])
    if "q_start" in ev and ev.q_start.notna().any():
        out["q_start_first"] = float(ev.q_start.dropna().iloc[0])
    return out


def group_curve(evals):
    """Mean and SE over seeds on the union of their evaluation steps."""
    frames = [e.set_index("online_steps").success_rate.rename(i) for i, e in enumerate(evals)]
    table = pd.concat(frames, axis=1).sort_index().interpolate(limit_area="inside")
    mean = table.mean(axis=1)
    n = table.notna().sum(axis=1)
    se = table.std(axis=1, ddof=1).fillna(0) / np.sqrt(n.clip(lower=1))
    return table.index.to_numpy(float), mean.to_numpy(float), se.to_numpy(float), n.to_numpy(int)


def binned_diagnostics(trains, bin_env=5000):
    """train_log rows binned on online steps and averaged over seeds."""
    frames = []
    for df in trains:
        if df is None:
            continue
        cols = [c for c in DIAG_COLS if c in df]
        d = df[["online_steps"] + cols].copy()
        d["bin"] = (d.online_steps // bin_env) * bin_env
        frames.append(d.groupby("bin")[cols].mean())
    if not frames:
        return None
    wide = pd.concat(frames, axis=1)
    return wide.T.groupby(level=0).mean().T.sort_index()


# ------------------------------------------------------------------ figures

def plot_success(ax, runs, groups, reference=None, smooth=0, xmax=None):
    for i, g in enumerate(groups):
        if g not in runs:
            continue
        evals = [load_eval(p) for p in runs[g].values()]
        x, m, se, n = group_curve(evals)
        if xmax is not None:
            keep = x <= xmax
            x, m, se, n = x[keep], m[keep], se[keep], n[keep]
        if smooth > 1:
            k = np.ones(smooth) / smooth
            m = np.convolve(m, k, mode="same")
        n_label = "n=%d" % max(n) if min(n) == max(n) else "n=%d-%d" % (min(n), max(n))
        color = "C%d" % i
        # lines and markers above every band, so a curve inside another's
        # band (mix_linear inside mix_fixed's) stays visible
        ax.fill_between(x, m - se, m + se, alpha=0.15, color=color, zorder=1, lw=0)
        ax.plot(x, m, label="%s (%s)" % (g, n_label), color=color, ls=LINESTYLES[i % len(LINESTYLES)],
                marker=MARKERS[i % len(MARKERS)], ms=3.5, lw=1.5, zorder=3)
    if reference is not None:
        ax.axhline(reference[0], color="k", ls="--", lw=1, label="pi_dp, N(0,I) noise", zorder=2)
    ax.set_xlabel("online environment steps")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    if xmax is not None:
        ax.set_xlim(-1000, xmax + 1000)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_diagnostics(fig_axes, runs, groups):
    # Distinct line styles: conditions whose actor statistics coincide
    # (fixalpha and warmupc on Can) would otherwise hide each other.
    for i, g in enumerate(groups):
        if g not in runs:
            continue
        diag = binned_diagnostics([load_train(p) for p in runs[g].values()])
        if diag is None:
            continue
        for ax, col in zip(fig_axes, DIAG_COLS):
            if col in diag:
                ax.plot(diag.index, diag[col], label=g, ls=LINESTYLES[i % len(LINESTYLES)], lw=1.6)
    for ax, col in zip(fig_axes, DIAG_COLS):
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("online environment steps", fontsize=8)
        ax.grid(alpha=0.3)
        if col in ("ent_coef", "ratio_ge_gq", "gq_norm"):
            ax.set_yscale("log")
    fig_axes[0].legend(fontsize=7)


def plot_qgap(ax, runs, groups):
    drew = False
    for g in groups:
        if g not in runs:
            continue
        frames = []
        for p in runs[g].values():
            ev = load_eval(p)
            if "q_start" in ev and "mc_return" in ev and ev.q_start.notna().any():
                frames.append((ev.q_start - ev.mc_return).rename(p).set_axis(ev.online_steps))
        if frames:
            t = pd.concat(frames, axis=1).sort_index()
            ax.plot(t.index, t.mean(axis=1), label=g)
            drew = True
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("online environment steps")
    ax.set_ylabel("Q_W(s0) - discounted return")
    ax.grid(alpha=0.3)
    if drew:
        ax.legend(fontsize=8)
    return drew


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--axes", default="", help="name=group1,group2;name2=... (default: DEFAULT_AXES)")
    ap.add_argument("--until_env", type=int, default=100000,
                    help="end of the early window in env steps; online cut-off = until_env - rollout")
    ap.add_argument("--smooth", type=int, default=0, help="moving-average window in evaluation points")
    ap.add_argument("--levels", default="0.5,0.8", help="success levels for the time-to-level columns (t50, t80)")
    args = ap.parse_args()
    levels = tuple(float(v) for v in args.levels.split(",") if v)
    level_keys = ["t%d" % int(round(v * 100)) for v in levels]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = find_runs(args.logs)
    if not runs:
        raise SystemExit("no <task>_<group>_s<seed> directories with eval_log.csv under %s" % args.logs)
    os.makedirs(args.out, exist_ok=True)
    axes = dict(DEFAULT_AXES)
    if args.axes:
        axes = {}
        for part in args.axes.split(";"):
            name, groups = part.split("=")
            axes[name] = groups.split(",")
    references = {task: base_reference(args.logs, task) for task in ROLLOUT_ENV}
    until = {task: args.until_env - r for task, r in ROLLOUT_ENV.items()}

    rows = []
    for g, seeds in sorted(runs.items()):
        task = group_task(g)
        reference = references.get(task)
        per_seed = []
        for s, p in sorted(seeds.items()):
            ev = load_eval(p)
            ev.attrs["path"] = p
            m = run_metrics(ev, until[task], reference, levels=levels)
            m.update({"group": g, "seed": s, "run": os.path.basename(p), "until_online": until[task]})
            rows.append(m)
            per_seed.append(m)
        agg = {"group": g, "seed": "mean", "run": "n=%d" % len(per_seed), "until_online": until[task]}
        for key in ("step0", "min_in_window", "min_at", "dip_depth", "recovery_at", "recovery_pi_dp_at",
                    "auc_early", "at_env129k", "final", "mc_return_first", "q_start_first") + tuple(level_keys):
            vals = [m[key] for m in per_seed if key in m and np.isfinite(m[key])]
            agg[key] = float(np.mean(vals)) if vals else np.nan
            agg[key + "_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
            agg[key + "_n"] = len(vals)
        # the minimum of the seed-mean curve inside the window (the number the
        # handoff tables call "lowest"), next to the mean of per-seed minima
        # Read on the common 5k grid (online 5,008 x k, the evaluation steps
        # of a 5k-schedule run) so runs evaluated every 2.5k do not get a
        # deeper minimum just from being sampled more often.
        x, m, _, _ = group_curve([load_eval(p) for p in seeds.values()])
        grid = np.arange(GRID_ONLINE, min(until[task], x[-1]) + 1, GRID_ONLINE)
        if len(grid):
            mg = np.interp(grid, x, m)
            j = int(np.argmin(mg))
            agg["min_mean_curve"], agg["min_mean_curve_at"] = float(mg[j]), int(grid[j])
        if reference is not None and np.isfinite(agg["auc_early"]):
            agg["regret_vs_pi_dp"] = reference[0] - agg["auc_early"]
        rows.append(agg)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    # The summary in the form time-to-level / AUC / final, mean +- SE over seeds.
    # A time-to-level averages the seeds that reached the level; "k/n" says how many did.
    means = metrics[metrics.seed == "mean"]
    summary = pd.DataFrame({
        "group": means.group, "n": means.run.str.replace("n=", ""),
        **{k: ["%.0f (%d/%s)" % (t, c, n) if np.isfinite(t) else "never (0/%s)" % n
               for t, c, n in zip(means[k], means[k + "_n"], means.run.str.replace("n=", ""))] for k in level_keys},
        "auc_early": ["%.3f+-%.3f" % ab for ab in zip(means.auc_early, means.auc_early_se.fillna(0))],
        "final": ["%.3f+-%.3f" % ab for ab in zip(means.final, means.final_se.fillna(0))],
        "at_env129k": ["%.3f" % v if np.isfinite(v) else "-" for v in means.at_env129k],
    })
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print(summary.to_string(index=False))
    for task, reference in references.items():
        if reference is not None:
            print("pi_dp reference (%s): %.3f +- %.3f" % ((task,) + reference))
    print("early window (online steps): %s" % ", ".join("%s 0-%d" % kv for kv in until.items()))

    for name, groups in axes.items():
        present = [g for g in groups if g in runs]
        if not present:
            continue
        task = group_task(present[0])
        reference = references.get(task)
        for suffix, xmax in (("", None), ("_early", until[task])):
            fig, ax = plt.subplots(figsize=(7, 4))
            plot_success(ax, runs, present, reference, args.smooth, xmax)
            ax.set_title("axis: %s" % name + ("  (early window, online 0-%dk)" % (xmax // 1000) if xmax else ""))
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "success_%s%s.png" % (name, suffix)), dpi=150)
            plt.close(fig)

        fig, fig_axes = plt.subplots(3, 3, figsize=(14, 9))
        fig_axes = fig_axes.ravel()
        plot_diagnostics(fig_axes, runs, present)
        fig.suptitle("diagnostics, axis: %s" % name)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "diagnostics_%s.png" % name), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        if plot_qgap(ax, runs, present):
            ax.set_title("Q over-estimation, axis: %s (valid once ent_coef < 0.1)" % name)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "qgap_%s.png" % name), dpi=150)
        plt.close(fig)
    print("figures in", args.out)


if __name__ == "__main__":
    main()

"""Learning curves, dip metrics and diagnostics from the run CSVs.

    python scripts/plot_results.py --logs $PROJ/logs --out $PROJ/figures

Runs are grouped by name: can_<group>_s<seed>. Every group with at least one
seed gets a mean +- standard-error success curve; groups are drawn together
on one figure per "axis" (--axes, default: the critic axis and the mix axis).

Outputs in --out:
  success_<axis>.png       success rate vs env steps, mean +- SE over seeds
  diagnostics_<axis>.png   ent_coef, mu_absmean, w_absmean, w_frac_sat,
                           offline_p, qw_mean vs env steps (train_log.csv)
  qgap_<axis>.png          q_start - mc_return from eval_log.csv, where present
  metrics.csv              per run and per group: step0, min over the dip
                           window, dip depth, recovery step, AUC, final
                           (and regret against pi_dp when base_policy_eval.csv exists)

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
# Groups keep their Can names; other tasks are prefixed (square_baseline), and
# each axis is drawn on its own so tasks never share a y axis.
DEFAULT_AXES = {
    "critic": ["baseline", "warmup", "iql", "warmupc", "fixalpha"],
    "mix": ["baseline", "mix_prefill", "mix_fixed", "mix_linear", "iql_linear"],
    "square": ["square_baseline"],
}
DIAG_COLS = ["ent_coef", "mu_absmean", "w_absmean", "w_frac_sat", "log_std_mean", "offline_p", "qw_mean"]


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


def load_eval(path):
    df = pd.read_csv(os.path.join(path, "eval_log.csv"))
    if "deterministic" in df:
        df = df[df.deterministic == 0]
    df = df.drop_duplicates("env_steps", keep="last").sort_values("env_steps")
    return df.reset_index(drop=True)


def load_train(path):
    f = os.path.join(path, "train_log.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    return df.sort_values("env_steps").reset_index(drop=True)


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

def run_metrics(ev, dip_until=100000, final_points=3):
    """Dip statistics of one run's evaluation curve."""
    x, y = ev.env_steps.to_numpy(float), ev.success_rate.to_numpy(float)
    step0 = y[0] if len(y) and x[0] == 0 else np.nan
    win = (x > 0) & (x <= dip_until)
    out = {"step0": step0, "n_evals": int(len(y)), "last_env_steps": int(x[-1]) if len(x) else 0}
    if win.any():
        i_min = np.argmin(np.where(win, y, np.inf))
        out["min_in_window"] = float(y[i_min])
        out["min_at"] = int(x[i_min])
        out["dip_depth"] = float(step0 - y[i_min]) if np.isfinite(step0) else np.nan
        # first evaluation after the minimum that reaches the step-0 level again
        after = (x > x[i_min]) & (y >= step0)
        out["recovery_at"] = int(x[np.argmax(after)]) if after.any() and np.isfinite(step0) else np.nan
        xs, ys = x[x <= dip_until], y[x <= dip_until]
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        out["auc_window"] = float(trapezoid(ys, xs) / (xs[-1] - xs[0])) if len(xs) > 1 else np.nan
    out["final"] = float(np.mean(y[-final_points:])) if len(y) else np.nan
    # The last evaluation every budget reaches (150k runs stop before the
    # 154k evaluation), so groups with different budgets stay comparable.
    if len(x) > 1 and x[-1] >= 129000:
        out["at_129k"] = float(np.interp(129152, x, y))
    if "mc_return" in ev and ev.mc_return.notna().any():
        out["mc_return_first"] = float(ev.mc_return.dropna().iloc[0])
    if "q_start" in ev and ev.q_start.notna().any():
        out["q_start_first"] = float(ev.q_start.dropna().iloc[0])
    return out


def group_curve(evals):
    """Mean and SE over seeds on the union of their evaluation steps."""
    frames = [e.set_index("env_steps").success_rate.rename(i) for i, e in enumerate(evals)]
    table = pd.concat(frames, axis=1).sort_index().interpolate(limit_area="inside")
    mean = table.mean(axis=1)
    n = table.notna().sum(axis=1)
    se = table.std(axis=1, ddof=1).fillna(0) / np.sqrt(n.clip(lower=1))
    return table.index.to_numpy(float), mean.to_numpy(float), se.to_numpy(float), n.to_numpy(int)


def binned_diagnostics(trains, bin_env=5000):
    """train_log rows binned on env steps and averaged over seeds."""
    frames = []
    for df in trains:
        if df is None:
            continue
        cols = [c for c in DIAG_COLS if c in df]
        d = df[["env_steps"] + cols].copy()
        d["bin"] = (d.env_steps // bin_env) * bin_env
        frames.append(d.groupby("bin")[cols].mean())
    if not frames:
        return None
    wide = pd.concat(frames, axis=1)
    return wide.T.groupby(level=0).mean().T.sort_index()


# ------------------------------------------------------------------ figures

def plot_success(ax, runs, groups, reference=None, smooth=0):
    for g in groups:
        if g not in runs:
            continue
        evals = [load_eval(p) for p in runs[g].values()]
        x, m, se, n = group_curve(evals)
        if smooth > 1:
            k = np.ones(smooth) / smooth
            m = np.convolve(m, k, mode="same")
        n_label = "n=%d" % max(n) if min(n) == max(n) else "n=%d-%d" % (min(n), max(n))
        ax.plot(x, m, label="%s (%s)" % (g, n_label))
        ax.fill_between(x, m - se, m + se, alpha=0.2)
    if reference is not None:
        ax.axhline(reference[0], color="k", ls="--", lw=1, label="pi_dp, N(0,I) noise")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


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
        ax.grid(alpha=0.3)
        if col == "ent_coef":
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
                frames.append((ev.q_start - ev.mc_return).rename(p).set_axis(ev.env_steps))
        if frames:
            t = pd.concat(frames, axis=1).sort_index()
            ax.plot(t.index, t.mean(axis=1), label=g)
            drew = True
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("environment steps")
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
    ap.add_argument("--axes", default="", help="name=group1,group2;name2=... (default: critic and mix axes)")
    ap.add_argument("--dip_until", type=int, default=100000)
    ap.add_argument("--smooth", type=int, default=0, help="moving-average window in evaluation points")
    args = ap.parse_args()

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
    references = {task: base_reference(args.logs, task) for task in ("can", "square", "lift", "transport")}

    rows = []
    for g, seeds in sorted(runs.items()):
        per_seed = []
        for s, p in sorted(seeds.items()):
            m = run_metrics(load_eval(p), args.dip_until)
            m.update({"group": g, "seed": s, "run": os.path.basename(p)})
            rows.append(m)
            per_seed.append(m)
        agg = {"group": g, "seed": "mean", "run": "n=%d" % len(per_seed)}
        for key in ("step0", "min_in_window", "dip_depth", "recovery_at", "auc_window", "at_129k", "final",
                    "mc_return_first", "q_start_first"):
            vals = [m[key] for m in per_seed if key in m and np.isfinite(m[key])]
            agg[key] = float(np.mean(vals)) if vals else np.nan
            agg[key + "_se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        reference = references.get(group_task(g))
        if reference is not None and np.isfinite(agg["auc_window"]):
            agg["regret_vs_pi_dp"] = reference[0] - agg["auc_window"]
        rows.append(agg)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    print(metrics[metrics.seed == "mean"].to_string(index=False))
    for task, reference in references.items():
        if reference is not None:
            print("pi_dp reference (%s): %.3f +- %.3f" % ((task,) + reference))

    for name, groups in axes.items():
        present = [g for g in groups if g in runs]
        if not present:
            continue
        reference = references.get(group_task(present[0]))
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_success(ax, runs, present, reference, args.smooth)
        ax.set_title("axis: %s" % name)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "success_%s.png" % name), dpi=150)
        plt.close(fig)

        fig, fig_axes = plt.subplots(2, 4, figsize=(14, 6))
        fig_axes = fig_axes.ravel()
        plot_diagnostics(fig_axes, runs, present)
        fig_axes[-1].axis("off")
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

"""When does alpha (or the gradient ratio) cross a threshold, and when does the dip happen, per seed.

    cd scripts && python alpha_timing.py --logs $PROJ/logs --out $PROJ/figures [--groups baseline,alr_half,alr_double,rs_025,rs_2,hardq]

For every run in the groups, from train_log.csv: the env step at which ent_coef
first falls below 0.3 / 0.1 / 0.03, the env step at which the running median of
ratio_ge_gq (entropy gradient over Q gradient on the actor, logged since 21cef73)
first falls below 3 / 1 / 0.3, and the values of ent_coef, ratio_ge_gq and
qw_absmean around the first evaluation below the pi_dp reference. From the
evaluation curve smoothed with a 3-point moving average: the first evaluation
below the reference, the minimum inside the dip window, and the first evaluation
back above the reference after the minimum.

Writes alpha_timing.csv, alpha_timing.png (alpha crossing vs dip time) and, when
the ratio columns exist, ratio_timing.png (ratio crossing vs dip time, and the
ratio and alpha at the first drop per group). The judgment for the Q-scale
hypothesis is the spread of ratio_at_first_below across conditions against the
spread of alpha_at_first_below: the quantity that is constant at the moment of
the first drop is the switch.
"""

import argparse
import os

import numpy as np
import pandas as pd

from plot_results import base_reference, find_runs, load_eval, load_train, group_task

THRESHOLDS = (0.3, 0.1, 0.03)
RATIO_THRESHOLDS = (3.0, 1.0, 0.3)
AT_COLS = ("ent_coef", "ratio_ge_gq", "qw_absmean", "gq_norm", "ge_norm")


def crossing(train, threshold, col="ent_coef", k=1):
    """First env step at which the k-row running median of col is below threshold."""
    if train is None or col not in train:
        return np.nan
    v = train[col].astype(float)
    if k > 1:
        v = v.rolling(k, center=True, min_periods=1).median()
    below = train[v < threshold]
    return float(below.env_steps.iloc[0]) if len(below) else np.nan


def value_at(train, col, env_step, halfwidth=1500):
    """Median of a train_log column within +-halfwidth env steps of env_step."""
    if train is None or col not in train or env_step is None or np.isnan(env_step):
        return np.nan
    w = train[(train.env_steps >= env_step - halfwidth) & (train.env_steps <= env_step + halfwidth)]
    v = w[col].astype(float).dropna()
    return float(v.median()) if len(v) else np.nan


def smooth(y, k=3):
    if len(y) < k:
        return y.copy()
    out = y.copy()
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        out[i] = np.mean(y[lo:hi])
    return out


def dip_times(ev, train_start, reference, window_end=80000):
    x, y = ev.env_steps.to_numpy(float), ev.success_rate.to_numpy(float)
    ys = smooth(y)
    after = x > train_start
    out = {}
    below = after & (ys < reference)
    out["first_below_ref"] = float(x[below][0]) if below.any() else np.nan
    win = after & (x <= window_end)
    if win.any():
        i = int(np.argmin(np.where(win, ys, np.inf)))
        out["bottom_at"] = float(x[i])
        out["bottom_smoothed"] = float(ys[i])
        out["bottom_raw"] = float(y[i])
        back = (x > x[i]) & (ys >= reference)
        out["back_above_ref"] = float(x[back][0]) if back.any() else np.nan
    return out


def log_spread(values):
    """Std of log10 over the finite positive entries, and their count."""
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0], float)
    return (float(np.std(np.log10(v))), len(v)) if len(v) else (np.nan, 0)


def scatter_with_diagonal(ax, df, xcol, ycol, xlabel, title):
    d = df.dropna(subset=[xcol, ycol])
    for g, dg in d.groupby("group"):
        ax.scatter(dg[xcol], dg[ycol], label=g, s=40)
    if len(d):
        lo, hi = float(np.nanmin(d[xcol])) - 2000, float(np.nanmax(d[xcol])) + 2000
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("env step")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="alr_double,baseline,alr_half,rs_025,rs_2,hardq")
    ap.add_argument("--train_start", type=int, default=24016, help="env steps of the initial rollout")
    ap.add_argument("--ratio_k", type=int, default=3, help="rows in the running median of ratio_ge_gq")
    args = ap.parse_args()

    runs = find_runs(args.logs)
    rows = []
    for g in args.groups.split(","):
        for seed, path in sorted(runs.get(g, {}).items()):
            ev, tr = load_eval(path), load_train(path)
            ref = base_reference(args.logs, group_task(g))
            ref = ref[0] if ref else 0.4
            row = {"group": g, "seed": seed}
            for t in THRESHOLDS:
                row["alpha_below_%g" % t] = crossing(tr, t)
            for t in RATIO_THRESHOLDS:
                row["ratio_below_%g" % t] = crossing(tr, t, "ratio_ge_gq", k=args.ratio_k)
            row.update(dip_times(ev, args.train_start, ref))
            for col in AT_COLS:
                row["%s_at_first_below" % col] = value_at(tr, col, row.get("first_below_ref", np.nan))
            row["ratio_ge_gq_at_bottom"] = value_at(tr, "ratio_ge_gq", row.get("bottom_at", np.nan))
            rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "alpha_timing.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df.to_string(index=False, float_format=lambda v: "%.3g" % v))

    have_ratio = "ratio_ge_gq_at_first_below" in df and df["ratio_ge_gq_at_first_below"].notna().any()
    cols = ["alpha_below_0.1", "first_below_ref", "bottom_at", "back_above_ref"]
    if have_ratio:
        cols = ["alpha_below_0.1", "ratio_below_1", "first_below_ref", "bottom_at",
                "ent_coef_at_first_below", "ratio_ge_gq_at_first_below", "qw_absmean_at_first_below"]
    summary = df.groupby("group")[cols].agg(["mean", "std"])
    print("\nper group (mean, std):")
    print(summary.to_string(float_format=lambda v: "%.3g" % v))

    ok = df.dropna(subset=["alpha_below_0.1", "bottom_at"])
    if len(ok) >= 3:
        r_bottom = np.corrcoef(ok["alpha_below_0.1"], ok["bottom_at"])[0, 1]
        ok2 = df.dropna(subset=["alpha_below_0.1", "first_below_ref"])
        r_first = np.corrcoef(ok2["alpha_below_0.1"], ok2["first_below_ref"])[0, 1] if len(ok2) >= 3 else np.nan
        slope = np.polyfit(ok["alpha_below_0.1"], ok["bottom_at"], 1)[0]
        print("\ncorrelation(alpha<0.1 crossing, bottom) = %.2f, slope = %.2f env step per env step" % (r_bottom, slope))
        print("correlation(alpha<0.1 crossing, first eval below pi_dp) = %.2f" % r_first)

    if have_ratio:
        # Which quantity is constant when the first drop happens? Spread in
        # log10 across every run that has both; the smaller spread is the
        # better switch. A spread of 0.1 is a factor 1.26, 0.5 is a factor 3.
        print("\nat the first evaluation below pi_dp (spread = std of log10 over runs):")
        for col in AT_COLS:
            key = "%s_at_first_below" % col
            s, n = log_spread(df[key])
            vals = df[key].dropna()
            if len(vals):
                print("  %-12s n=%d  median %.3g  range [%.3g, %.3g]  spread %.2f"
                      % (col, n, vals.median(), vals.min(), vals.max(), s))
        ok3 = df.dropna(subset=["ratio_below_1", "first_below_ref"])
        if len(ok3) >= 3:
            r = np.corrcoef(ok3["ratio_below_1"], ok3["first_below_ref"])[0, 1]
            print("correlation(ratio<1 crossing, first eval below pi_dp) = %.2f (n=%d)" % (r, len(ok3)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(axes, ["first_below_ref", "bottom_at"], ["first evaluation below pi_dp", "bottom of the dip"]):
        scatter_with_diagonal(ax, df, "alpha_below_0.1", col, "env step at which alpha < 0.1", title)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "alpha_timing.png"), dpi=150)
    print("wrote", os.path.join(args.out, "alpha_timing.png"))

    if have_ratio:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        scatter_with_diagonal(axes[0], df, "ratio_below_1", "first_below_ref",
                              "env step at which ratio_ge_gq < 1", "first evaluation below pi_dp")
        groups = list(dict.fromkeys(df.group))
        for ax, col, title in zip(axes[1:], ["ratio_ge_gq_at_first_below", "ent_coef_at_first_below"],
                                  ["ratio_ge_gq at the first drop", "alpha at the first drop"]):
            for i, g in enumerate(groups):
                v = df[df.group == g][col].dropna()
                ax.scatter([i] * len(v), v, s=40)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, rotation=30, fontsize=8)
            ax.set_yscale("log")
            ax.set_title(title)
            ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "ratio_timing.png"), dpi=150)
        print("wrote", os.path.join(args.out, "ratio_timing.png"))


if __name__ == "__main__":
    main()

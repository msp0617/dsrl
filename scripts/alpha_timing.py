"""When does alpha cross a threshold, and when does the dip happen, per seed.

    python scripts/alpha_timing.py --logs $PROJ/logs --out $PROJ/figures [--groups baseline,alr_half,alr_double]

For every run in the groups: the env step at which ent_coef first falls below
0.3 / 0.1 / 0.03 (from train_log.csv), and from the evaluation curve smoothed
with a 3-point moving average: the first evaluation below the pi_dp reference,
the minimum inside the dip window, and the first evaluation back above the
reference after the minimum. Writes alpha_timing.csv and alpha_timing.png
(crossing vs dip time, one point per seed).
"""

import argparse
import os

import numpy as np
import pandas as pd

from plot_results import base_reference, find_runs, load_eval, load_train, group_task

THRESHOLDS = (0.3, 0.1, 0.03)


def crossing(train, threshold):
    if train is None or "ent_coef" not in train:
        return np.nan
    below = train[train.ent_coef < threshold]
    return float(below.env_steps.iloc[0]) if len(below) else np.nan


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="alr_double,baseline,alr_half")
    ap.add_argument("--train_start", type=int, default=24016, help="env steps of the initial rollout")
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
            row.update(dip_times(ev, args.train_start, ref))
            rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "alpha_timing.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    summary = df.groupby("group")[["alpha_below_0.1", "first_below_ref", "bottom_at", "back_above_ref"]].agg(["mean", "std"])
    print("\nper group (mean, std):")
    print(summary.to_string())

    ok = df.dropna(subset=["alpha_below_0.1", "bottom_at"])
    if len(ok) >= 3:
        r_bottom = np.corrcoef(ok["alpha_below_0.1"], ok["bottom_at"])[0, 1]
        ok2 = df.dropna(subset=["alpha_below_0.1", "first_below_ref"])
        r_first = np.corrcoef(ok2["alpha_below_0.1"], ok2["first_below_ref"])[0, 1] if len(ok2) >= 3 else np.nan
        slope = np.polyfit(ok["alpha_below_0.1"], ok["bottom_at"], 1)[0]
        print("\ncorrelation(alpha<0.1 crossing, bottom) = %.2f, slope = %.2f env step per env step" % (r_bottom, slope))
        print("correlation(alpha<0.1 crossing, first eval below pi_dp) = %.2f" % r_first)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(axes, ["first_below_ref", "bottom_at"], ["first evaluation below pi_dp", "bottom of the dip"]):
        for g, d in df.groupby("group"):
            ax.scatter(d["alpha_below_0.1"], d[col], label=g, s=40)
        lo = np.nanmin(df["alpha_below_0.1"]) - 2000
        hi = np.nanmax(df["alpha_below_0.1"]) + 2000
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
        ax.set_xlabel("env step at which alpha < 0.1")
        ax.set_ylabel("env step")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "alpha_timing.png"), dpi=150)
    print("wrote", os.path.join(args.out, "alpha_timing.png"))


if __name__ == "__main__":
    main()

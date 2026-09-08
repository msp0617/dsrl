"""replay_bars.png for the poster: early-online AUC and 129k success of the replay axis.

    python scripts/replay_bars.py --summary <figs>/summary.csv --out <figs>/replay_bars.png

Reads the summary.csv written by plot_results.py (columns group, n, auc_early
as "mean+-se", at_env129k) and draws two bar panels for the demo-replay
conditions. The figure keeps h/w below 0.478 so it fits the poster slot
(276 x 132 mm) that build_poster.py reserves for it.
"""

import argparse
import csv
import os

GROUPS = [  # (group in summary.csv, label on the axis)
    ("baseline", "baseline\nno demos"),
    ("mix_prefill", "prefill\n(upstream)"),
    ("mix_fixed", "fixed 0.5\n(RLPD)"),
    ("mix_linear", "linear\n0.8→0.1"),
    ("iql_prefill", "IQL critic\n+ prefill"),
]


def parse_pm(value):
    """'0.682+-0.019' -> (0.682, 0.019); '-' -> (nan, nan)."""
    if not value or value.strip() == "-":
        return float("nan"), float("nan")
    if "+-" in value:
        m, s = value.split("+-")
        return float(m), float(s)
    return float(value), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="can", help="only used in the title")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = {}
    with open(args.summary, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["group"]] = row
    missing = [g for g, _ in GROUPS if g not in rows]
    if missing:
        raise SystemExit("summary.csv has no rows for: %s" % ", ".join(missing))

    labels = [lab + "\nn=%s" % rows[g]["n"] for g, lab in GROUPS]
    auc = [parse_pm(rows[g]["auc_early"]) for g, _ in GROUPS]
    final = [parse_pm(rows[g]["at_env129k"]) for g, _ in GROUPS]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    x = range(len(GROUPS))
    colors = ["#888888"] + ["#1f77b4"] * (len(GROUPS) - 1)
    ax = axes[0]
    ax.bar(x, [m for m, _ in auc], yerr=[s for _, s in auc], color=colors, capsize=3)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("early-online AUC (online 0–76k)", fontsize=8)
    ax.set_ylim(0, 1); ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Early window, mean ± SE over seeds", fontsize=9)
    for i, (m, _) in enumerate(auc):
        ax.text(i, m + 0.03, "%.2f" % m, ha="center", fontsize=7)

    ax = axes[1]
    ax.bar(x, [m for m, _ in final], color=colors)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("success at env step 129k", fontsize=8)
    ax.set_ylim(0, 1); ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Final evaluation (mean over seeds)", fontsize=9)
    for i, (m, _) in enumerate(final):
        ax.text(i, m + 0.03, "%.2f" % m, ha="center", fontsize=7)

    fig.suptitle("Demo replay (%s): keeping demonstrations in the buffer is the largest lever" % args.task, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    w, h = fig.get_size_inches()
    print("wrote %s (h/w = %.3f, slot needs <= 0.478)" % (args.out, h / w))


if __name__ == "__main__":
    main()

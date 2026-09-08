"""Poster figures from the audited evaluation table (results/<date>/eval_clean.csv).

    python scripts/poster_figs.py --results results/2026-09-08 --out <dir>

Writes, sized for an A1 poster placed at 100 %:

  critic_ladder_early.png  two panels (Can, Square), early online window,
                           baseline / IQL / TD / CQL-style / Cal-QL-style
  dip_can.png              one panel, Can over the full horizon,
                           baseline vs fixed alpha 0.3 vs target entropy 12 (alpha0 0.3)

Curves are the seed mean with a +/- SE band on the union of the seeds' online
steps (linear interpolation inside each seed's range only), the same rule as
scripts/plot_completed_analysis.py. The dashed reference is the frozen
diffusion policy with N(0, I) noise (results/<date>/references.csv).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROLLOUT = {"can": 24016, "square": 32016}
END = {"can": 129152, "square": 127136}
EARLY = {"can": 75984, "square": 67984}
COLORS = {"baseline": "#687386", "iql": "#2563eb", "td": "#b7791f", "cql": "#8b5cf6",
          "calql": "#d43f57", "fixalpha_03": "#c9368c", "tent12i": "#6d4c41"}


def mean_curve(data, task, group, maximum):
    df = data[(data.task == task) & (data.group == group)]
    frames = [d.set_index("online_steps").success_rate.rename(s) for s, d in df.groupby("seed")]
    table = pd.concat(frames, axis=1).sort_index()
    if maximum not in table.index:
        table = table.reindex(table.index.union([maximum])).sort_index()
    table = table.interpolate(method="index", limit_area="inside")
    table = table.loc[(table.index <= maximum) & table.notna().all(axis=1)]
    return table.index.to_numpy() / 1000, table.mean(axis=1).to_numpy(), table.sem(axis=1).to_numpy(), len(frames)


def draw(ax, data, task, groups, labels, reference, maximum, lw=3.0):
    for group in groups:
        x, mean, se, n = mean_curve(data, task, group, maximum)
        ax.plot(x, mean, color=COLORS[group], lw=lw, marker="o", ms=4.5, label=f"{labels[group]} (n={n})")
        ax.fill_between(x, mean - se, mean + se, color=COLORS[group], alpha=.13, linewidth=0)
    ax.axhline(reference, color="#263238", lw=2, linestyle="--", label=f"frozen policy {reference:.3f}")
    ax.set_xlim(0, maximum / 1000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("online environment steps (k)")
    ax.grid(alpha=.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.results / "eval_clean.csv")
    refs = pd.read_csv(args.results / "references.csv")
    ref = {r.task: float(r.pi_dp_success) for r in refs.itertuples()} if "pi_dp_success" in refs else None
    if ref is None:  # fall back to the column names the audit wrote
        col = [c for c in refs.columns if "success" in c or "mean" in c][0]
        ref = {r.task: float(getattr(r, col)) for r in refs.itertuples()}

    plt.rcParams.update({"font.size": 17, "axes.titlesize": 20, "axes.labelsize": 17,
                         "legend.fontsize": 14, "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "font.family": "DejaVu Sans"})

    # 1) critic ladder, early window
    ladder = ["baseline", "iql", "td", "cql", "calql"]
    labels = {"baseline": "DSRL baseline", "iql": "IQL", "td": "TD", "cql": "CQL-style", "calql": "Cal-QL-style"}
    fig, axes = plt.subplots(1, 2, figsize=(10.9, 5.6))
    for ax, task, title in zip(axes, ("can", "square"), ("Can", "Square")):
        draw(ax, data, task, ladder, labels, ref[task], EARLY[task])
        ax.set_title(title, fontweight="bold")
    axes[0].set_ylabel("success rate")
    handles, names = axes[0].get_legend_handles_labels()
    names = [n.replace(" (n=5)", "").replace(" (n=3)", "") for n in names[:-1]] + ["frozen diffusion policy (Can 0.405, Square 0.494)"]
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(args.out / "critic_ladder_early.png", dpi=220, facecolor="white")
    fig.savefig(args.out / "critic_ladder_early.pdf", facecolor="white")
    plt.close(fig)

    # 2) Can dip removal
    groups = ["baseline", "fixalpha_03", "tent12i"]
    labels = {"baseline": "DSRL: auto-α, target ent. 0",
              "fixalpha_03": "fixed α = 0.3", "tent12i": "target ent. 12, α₀ = 0.3"}
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    draw(ax, data, "can", groups, labels, ref["can"], END["can"] - ROLLOUT["can"], lw=3.2)
    ax.set_ylabel("success rate")
    ax.legend(loc="lower right", framealpha=.95, fontsize=11.5, handlelength=1.5, borderaxespad=.4)
    fig.savefig(args.out / "dip_can.png", dpi=220, facecolor="white")
    fig.savefig(args.out / "dip_can.pdf", facecolor="white")
    plt.close(fig)
    print("wrote", sorted(p.name for p in args.out.iterdir()))


if __name__ == "__main__":
    main()

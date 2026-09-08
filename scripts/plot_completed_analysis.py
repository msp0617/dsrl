"""Plot the audited CSV outputs; never reads or modifies original logs.

python scripts/plot_completed_analysis.py --results results/2026-09-08
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROLLOUT = {"can": 24016, "square": 32016}
END = {"can": 129152, "square": 127136}
LABELS = {"baseline": "Baseline", "iql": "IQL", "td": "TD", "cql": "CQL-style",
          "calql": "Cal-QL-style", "calql_t12i": "Cal-QL + T12/init .3",
          "calql_prefill": "Cal-QL + prefill", "mix_prefill": "Prefill",
          "iql_prefill": "IQL + prefill", "fixalpha_03": "Fixed alpha .3",
          "tent12i": "T12/init .3", "tent12": "Target entropy 12",
          "tent12_hq": "Target 12 + hard backup", "warmupc": "Warmup critic",
          "warmup": "Warmup actor+critic"}
COLORS = {"baseline": "#687386", "iql": "#2563eb", "td": "#b7791f", "cql": "#8b5cf6",
          "calql": "#d43f57", "mix_prefill": "#087f5b", "iql_prefill": "#2f9e8d",
          "calql_prefill": "#e67700", "fixalpha_03": "#c9368c", "tent12i": "#6d4c41",
          "calql_t12i": "#0097b2", "tent12": "#d9480f", "tent12_hq": "#008080"}


def curves(ax, data, task, groups, reference, early=False):
    maximum = (100000 if early else END[task]) - ROLLOUT[task]
    for i, group in enumerate(groups):
        df = data[(data.task == task) & (data.group == group)]
        frames = [d.set_index("online_steps").success_rate.rename(s) for s, d in df.groupby("seed")]
        table = pd.concat(frames, axis=1).sort_index()
        if maximum not in table.index:
            table = table.reindex(table.index.union([maximum])).sort_index()
        table = table.interpolate(method="index", limit_area="inside")
        table = table.loc[(table.index <= maximum) & table.notna().all(axis=1)]
        x = table.index.to_numpy()/1000
        mean, se = table.mean(axis=1).to_numpy(), table.sem(axis=1).to_numpy()
        color = COLORS.get(group, f"C{i}")
        ax.plot(x, mean, color=color, lw=1.8, marker="o", ms=2.5,
                label=f"{LABELS.get(group, group)} (n={len(frames)})")
        ax.fill_between(x, mean-se, mean+se, color=color, alpha=.10, linewidth=0)
    ax.axhline(reference, color="#263238", lw=1, linestyle="--", label=f"Frozen diffusion: {reference:.3f}")
    ax.set(xlim=(0, maximum/1000), ylim=(0, 1), xlabel="Online environment steps (k)", ylabel="Success rate")
    ax.grid(alpha=.18)
    ax.legend(fontsize=8, loc="best", framealpha=.9)


def save(fig, out, name):
    fig.savefig(out/f"{name}.png", dpi=175, facecolor="white")
    fig.savefig(out/f"{name}.pdf", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()
    out = args.results
    data = pd.read_csv(out/"eval_clean.csv")
    groups = pd.read_csv(out/"groups.csv")
    seeds = pd.read_csv(out/"per_seed.csv")
    refs = pd.read_csv(out/"references.csv").set_index("task")["mean"].to_dict()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "pdf.fonttype": 42})
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9), layout="constrained")
    for i, task in enumerate(ROLLOUT):
        for j, early in enumerate((True, False)):
            curves(axs[i, j], data, task, ["baseline", "iql", "td", "cql", "calql"], refs[task], early)
            axs[i, j].set_title(f"{task.title()}: {'early window' if early else 'matched late horizon'}")
    fig.suptitle("Critic initialization reduces the early dip; Cal-QL is not uniformly best\nMean ± seed SE; CQL / Cal-QL are this project's critic-only adaptations", fontsize=13)
    save(fig, out, "critic_ladder")

    panels = [("can", ["baseline", "mix_prefill", "iql_prefill", "calql_prefill"], "Can: replay and critic combinations"),
              ("can", ["baseline", "fixalpha_03", "tent12i", "calql_t12i"], "Can: entropy control"),
              ("square", ["baseline", "iql", "mix_prefill", "calql"], "Square: late performance"),
              ("square", ["baseline", "tent12", "tent12_hq"], "Square: target-entropy / hard-backup intervention")]
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9), layout="constrained")
    for ax, (task, gs, title) in zip(axs.flat, panels):
        curves(ax, data, task, gs, refs[task]); ax.set_title(title)
    fig.suptitle("Early safety and late performance are different objectives\nMean ± seed SE; horizontal reference is the task-specific frozen diffusion policy", fontsize=13)
    save(fig, out, "recipes_and_tradeoffs")

    for task in ROLLOUT:
        tab = groups[groups.task == task].sort_values("auc_early", ascending=True)
        fig, axs = plt.subplots(1, 2, figsize=(13, max(5, len(tab)*.31+1.5)), sharey=True, layout="constrained")
        for ax, key, title in zip(axs, ["auc_early", "endpoint"], ["Early normalized AUC", f"Success at env {END[task]:,}"]):
            for i, row in enumerate(tab.itertuples()):
                ss = seeds[(seeds.task == task) & (seeds.group == row.group)][key].dropna()
                ax.scatter(ss, np.full(len(ss), i), color="#aeb8c7", s=12, zorder=2)
                m, se = getattr(row, key), getattr(row, key+"_se")
                if np.isfinite(m):
                    ax.errorbar(m, i, xerr=se if np.isfinite(se) else None, fmt="o", ms=5,
                                color=COLORS.get(row.group, "#2563eb"), capsize=3, zorder=3)
                    if key == "endpoint" and row.endpoint_n < row.n:
                        ax.text(.98, i, f"{row.endpoint_n}/{row.n}", ha="right", va="center", fontsize=8, color="#7a4400")
                else:
                    ax.text(.50, i, "No matched evaluation", ha="center", va="center", fontsize=8, color="#7a4400")
            ax.set(title=title, xlim=(0, 1), xlabel="Mean ± seed SE; pale dots = individual seeds")
            ax.grid(axis="x", alpha=.2)
            ax.axvline(refs[task], color="#263238", linestyle="--", lw=.8)
        axs[0].set_yticks(range(len(tab)), [f"{LABELS.get(r.group, r.group)}  (n={r.n})" for r in tab.itertuples()])
        fig.suptitle(f"{task.title()}: all {len(tab)} conditions, {int(tab.n.sum())} runs\nSorted by early AUC, not a universal ranking", fontsize=13)
        save(fig, out, f"all_conditions_{task}")

    diag = pd.read_csv(out/"diagnostic_groups.csv")
    columns = [("ent_coef", "Entropy coefficient alpha"), ("entropy_estimate", "Noise-policy entropy estimate: -log pi"),
               ("qw_mean", "Replay-state Q_W mean (soft value)"), ("w_frac_sat", "Saturated-noise fraction"),
               ("mu_absmean", "Actor mean absolute magnitude"), ("offline_p", "Offline replay fraction")]
    for task, gs in [("can", ["baseline", "calql", "tent12i", "calql_t12i", "calql_prefill"]),
                     ("square", ["baseline", "td", "calql", "tent12", "tent12_hq"])]:
        fig, axs = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
        for ax, (key, title) in zip(axs.flat, columns):
            for i, group in enumerate(gs):
                d = diag[(diag.task == task) & (diag.group == group) & (diag.bin_online <= 95000)]
                if key+"_mean" not in d:
                    continue
                color = COLORS.get(group, f"C{i}")
                ax.plot((d.bin_online+2500)/1000, d[key+"_mean"], color=color, lw=1.7, label=LABELS.get(group, group))
                ax.fill_between((d.bin_online+2500)/1000, d[key+"_mean"]-d[key+"_sem"], d[key+"_mean"]+d[key+"_sem"], color=color, alpha=.1)
            ax.set(title=title, xlabel="Online environment steps (k)")
            if key == "ent_coef":
                ax.set_yscale("log")
            ax.grid(alpha=.18)
        axs[0, 0].legend(fontsize=8)
        fig.suptitle(f"{task.title()} diagnostics: 5k bins, equal seed weighting\nQ_W includes entropy-bonus effects; magnitude alone is not a calibrated return error", fontsize=13)
        save(fig, out, f"diagnostics_{task}")
    print(f"Saved 6 figure pairs (PNG + PDF) to {out}")


if __name__ == "__main__":
    main()

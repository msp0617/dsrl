"""Audited, CPU-only analysis of the Can/Square CSVs (no training imports).

Raw logs are read-only. Outputs retain seed counts, actual endpoint steps,
restart audit, legacy AUC and boundary-interpolation sensitivity separately.
Run: python scripts/analyze_completed_runs.py --logs <logs> --out <results>
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROLLOUT = {"can": 24016, "square": 32016}
ENDPOINT = {"can": 129152, "square": 127136}
GRID = 5008
PATTERN = re.compile(r"^(can|square)_(.+)_s(\d+)$")
DIAGNOSTICS = ["ent_coef", "logp_mean", "qw_mean", "w_absmean",
               "mu_absmean", "w_frac_sat", "log_std_mean", "offline_p",
               "ratio_ge_gq", "gq_norm"]


def clean_history(raw):
    """Append-order overwrite: a rollback invalidates its old future branch.

    Do not sort before this operation. For ordinary same-step duplicates this
    equals keep-last. At a new step 0 the earlier attempt is discarded fully.
    No stochastic/deterministic histories are mixed by the caller.
    """
    keep = []
    steps = raw.env_steps.to_numpy()
    for i, step in enumerate(steps):
        while keep and steps[keep[-1]] >= step:
            keep.pop()
        keep.append(i)
    return raw.iloc[keep].copy().reset_index(drop=True)


def sem(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan


def interpolate(x, y, at):
    """Linear in numeric step distance, and never extrapolate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    at = np.asarray(at, float)
    return np.interp(at, x, y, left=np.nan, right=np.nan)


def window_curve(x, y, until, boundary="hold"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    inside = x <= until
    xs, ys = x[inside], y[inside]
    if len(xs) < 2 or xs[0] != 0:
        return np.array([]), np.array([])
    if xs[-1] < until:
        last = ys[-1] if boundary == "hold" else interpolate(x, y, until)
        if not np.isfinite(last):
            return np.array([]), np.array([])
        xs, ys = np.append(xs, until), np.append(ys, last)
    return xs, ys


def area(x, y, until, boundary="hold"):
    xs, ys = window_curve(x, y, until, boundary)
    return float(np.trapezoid(ys, xs) / until) if len(xs) else np.nan


def deficit_area(x, y, until, reference):
    """Integral of positive (reference-success), including zero crossings."""
    xs, ys = window_curve(x, y, until)
    if not len(xs):
        return np.nan
    d = reference - ys
    crossing = np.flatnonzero(d[:-1] * d[1:] < 0)
    roots = [xs[i] + (xs[i + 1] - xs[i]) * d[i] / (d[i] - d[i + 1])
             for i in crossing]
    xx = np.sort(np.concatenate([xs, roots]))
    return float(np.trapezoid(np.maximum(reference - np.interp(xx, xs, ys), 0), xx) / until)


def actual_eval(ev, target, tolerance=32):
    distance = (ev.env_steps - target).abs()
    i = distance.idxmin()
    if distance.loc[i] > tolerance:
        return np.nan, np.nan
    return float(ev.loc[i, "success_rate"]), int(ev.loc[i, "env_steps"])


def reference_rows(logs):
    rows = []
    for task in ROLLOUT:
        name = "base_policy_eval.csv" if task == "can" else "base_policy_eval_square.csv"
        path = logs / name
        if not path.exists():
            raise ValueError(f"Missing task-specific reference: {path}")
        raw = pd.read_csv(path)
        df = raw.drop_duplicates("seed", keep="last")
        rates = df.success_rate.to_numpy()
        pooled = float(np.average(rates, weights=df.episodes))
        rows.append(dict(task=task, n=len(df), episodes=int(df.episodes.sum()),
                         mean=float(np.mean(rates)), seed_se=sem(rates),
                         pooled_rate=pooled,
                         binomial_se=float(np.sqrt(pooled * (1-pooled) / df.episodes.sum())),
                         source_file=name, duplicate_seeds_removed=len(raw)-len(df)))
    return pd.DataFrame(rows)


def read_run(path, task, group, seed, reference):
    info = dict(run=path.name, task=task, group=group, seed=seed)
    audit = dict(info)
    frames = {}
    for kind in ("eval", "train"):
        file = path / f"{kind}_log.csv"
        raw = pd.read_csv(file)
        audit[kind + "_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        audit[kind + "_raw_rows"] = len(raw)
        if kind == "eval" and "deterministic" in raw:
            audit["deterministic_rows_excluded"] = int((raw.deterministic != 0).sum())
            raw = raw[raw.deterministic == 0].copy()
        if not len(raw) or raw.env_steps.isna().any():
            raise ValueError(f"Empty or invalid step column: {file}")
        audit[kind + "_duplicate_steps"] = int(raw.env_steps.duplicated().sum())
        audit[kind + "_backward_jumps"] = int((raw.env_steps.diff() < 0).sum())
        cleaned = clean_history(raw)
        audit[kind + "_removed_rows"] = len(raw) - len(cleaned)
        audit[kind + "_clean_rows"] = len(cleaned)
        audit[kind + "_last_env"] = int(cleaned.env_steps.iloc[-1])
        audit[kind + "_last_wall_time"] = str(cleaned.wall_time.iloc[-1])
        legacy = raw.drop_duplicates("env_steps", keep="last").sort_values("env_steps")
        audit[kind + "_legacy_extra_rows"] = len(legacy) - len(cleaned)
        cleaned["online_steps"] = np.where(cleaned.env_steps == 0, 0,
                                           cleaned.env_steps - ROLLOUT[task])
        if (cleaned.online_steps < 0).any():
            raise ValueError(f"Unexpected rollout/step mapping: {file}")
        frames[kind] = cleaned
    ev, tr = frames["eval"], frames["train"]
    if ev.success_rate.isna().any() or not ev.success_rate.between(0, 1).all():
        raise ValueError(f"Invalid success rate: {path}")
    if ev.env_steps.iloc[0] != 0:
        raise ValueError(f"Missing step 0: {path}")
    x, y = ev.online_steps.to_numpy(float), ev.success_rate.to_numpy(float)
    until = 100000 - ROLLOUT[task]
    # Max small tail hold is reported; a genuinely incomplete early trajectory
    # is never extended silently over tens of thousands of steps.
    inside = (x > 0) & (x <= until)
    tail = float(until - x[x <= until][-1])
    if tail > GRID:
        raise ValueError(f"Insufficient early-window coverage: {path}, tail={tail}")
    grid = np.arange(GRID, until + 1, GRID)
    yg = interpolate(x, y, grid)
    if np.isnan(yg).any():
        raise ValueError(f"Incomplete common early grid: {path}")
    imin = np.argmin(np.where(inside, y, np.inf))
    after_min = (x > x[imin]) & (y >= reference)
    endpoint, endpoint_step = actual_eval(ev, ENDPOINT[task])
    at10k, at10k_step = actual_eval(ev, ROLLOUT[task] + 10016)
    early_last = ev[ev.env_steps <= 100000].iloc[-1]
    metric = dict(info, n_evals=len(ev), first_success=y[0],
                  early_until_online=until, early_tail_held_steps=tail,
                  auc_early=area(x, y, until),
                  auc_early_boundary_linear=area(x, y, until, "linear"),
                  mean_deficit_pi_dp=deficit_area(x, y, until, reference),
                  min_observed=float(y[imin]), min_observed_online=int(x[imin]),
                  dip_from_start=float(y[0] - y[imin]),
                  min_grid=float(yg.min()),
                  no_dip_observed=bool(y[inside].min() >= reference),
                  no_dip_grid=bool(yg.min() >= reference),
                  recovery_pi_dp=float(x[after_min][0]) if after_min.any() else np.nan,
                  at10k=at10k, at10k_actual_env=at10k_step,
                  pre100k_success=float(early_last.success_rate),
                  pre100k_actual_env=int(early_last.env_steps),
                  endpoint=endpoint, endpoint_actual_env=endpoint_step,
                  endpoint_target_env=ENDPOINT[task],
                  at125k_interpolated=float(interpolate(ev.env_steps, y, 125000)),
                  final_last3_unmatched=float(y[-3:].mean()),
                  last_eval_env=int(ev.env_steps.iloc[-1]),
                  last_train_env=int(tr.env_steps.iloc[-1]))
    for threshold in (.5, .8):
        eligible = (x > 0) & (x <= ENDPOINT[task] - ROLLOUT[task]) & (y >= threshold)
        metric[f"t{int(threshold * 100)}_post_update"] = float(x[eligible][0]) if eligible.any() else np.nan
    # Per-seed temporal bins, then across-seed means. Logging frequency never
    # gives a seed extra weight in the group-level diagnostic.
    cols = [c for c in DIAGNOSTICS if c in tr]
    binned = tr.assign(bin_online=(tr.online_steps // 5000) * 5000).groupby("bin_online")[cols].mean().reset_index()
    for k, v in info.items():
        binned[k] = v
        ev[k] = v
    if "logp_mean" in binned:
        binned["entropy_estimate"] = -binned.logp_mean
    # Logit-space differential entropy estimate, not entropy of physical actions.
    return metric, audit, ev, binned, yg


def aggregate(metrics, curves):
    rows = []
    keys = ["first_success", "auc_early", "auc_early_boundary_linear",
            "mean_deficit_pi_dp", "min_observed", "dip_from_start", "min_grid",
            "at10k", "pre100k_success", "endpoint", "at125k_interpolated",
            "recovery_pi_dp", "t50_post_update", "t80_post_update"]
    for (task, group), df in metrics.groupby(["task", "group"], sort=True):
        row = dict(task=task, group=group, n=len(df),
                   seeds=",".join(str(s) for s in sorted(df.seed)),
                   endpoint_target_env=ENDPOINT[task],
                   early_until_online=100000 - ROLLOUT[task])
        for key in keys:
            vals = df[key].dropna()
            row[key] = float(vals.mean()) if len(vals) else np.nan
            row[key + "_se"] = sem(vals)
            row[key + "_n"] = len(vals)
        mean_grid = np.mean([curves[r] for r in df.run], axis=0)
        j = int(np.argmin(mean_grid))
        row["min_mean_curve"] = float(mean_grid[j])
        row["min_mean_curve_online"] = (j+1)*GRID
        row["min_mean_curve_se_at_selected_point"] = sem([curves[r][j] for r in df.run])
        row["no_dip_observed_n"] = int(df.no_dip_observed.sum())
        row["no_dip_grid_n"] = int(df.no_dip_grid.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def contrasts(metrics):
    """Matched seed-ID descriptive sensitivity; not paired identical rollouts."""
    rows = []
    pairs = [(t, g, "baseline") for t, g in metrics[["task", "group"]].drop_duplicates().itertuples(index=False)
             if g != "baseline"]
    pairs += [("can", "calql", "iql"), ("can", "calql", "cql"),
              ("can", "calql_t12i", "calql"), ("can", "calql_t12i", "tent12i"),
              ("can", "calql_prefill", "calql"), ("can", "calql_prefill", "mix_prefill"),
              ("square", "calql", "iql"), ("square", "calql", "cql"),
              ("square", "calql", "td"), ("square", "tent12_hq", "tent12"),
              ("square", "tent12_hq", "iql"), ("square", "tent12_hq", "mix_prefill")]
    for task, a, b in pairs:
        left = metrics[(metrics.task == task) & (metrics.group == a)].set_index("seed")
        right = metrics[(metrics.task == task) & (metrics.group == b)].set_index("seed")
        common = sorted(set(left.index) & set(right.index))
        for key in ("auc_early", "endpoint", "at10k", "mean_deficit_pi_dp"):
            delta = (left.loc[common, key] - right.loc[common, key]).dropna()
            rows.append(dict(task=task, treatment=a, comparator=b, metric=key,
                             n=len(delta), seeds=",".join(str(s) for s in delta.index),
                             delta=float(delta.mean()) if len(delta) else np.nan,
                             delta_se=sem(delta), positive_n=int((delta > 0).sum()),
                             negative_n=int((delta < 0).sum()),
                             per_seed_deltas=json.dumps({str(k): float(v) for k, v in delta.items()})))
    return pd.DataFrame(rows)


def fmt(mean, se):
    if not np.isfinite(mean):
        return "—"
    return f"{mean:.3f} ± {se:.3f}" if np.isfinite(se) else f"{mean:.3f} (n=1)"


def write_tables(groups, out):
    lines = ["# 전체 121 run 결과표", "", "성공률/AUC는 0–1 단위, ±는 seed 간 표본 SE. "
             "최저는 초기 공통 5,008 online-step 격자에서 seed-평균곡선의 최저값. "
             "후반은 Can 129,152 / Square 127,136 env step 실제 평가(±32 허용). "
             "미평가 값을 0으로 채우거나 외삽하지 않음.", ""]
    for task in ROLLOUT:
        lines += [f"## {task.title()}", "", "| 조건 | n | 초기 AUC | 평균곡선 최저 (online k) | 후반 성공률 | 후반 n | 기준선 아래로 안 간 seed¹ |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in groups[groups.task == task].itertuples():
            lines.append(f"| {r.group} | {r.n} | {fmt(r.auc_early, r.auc_early_se)} | "
                         f"{r.min_mean_curve:.3f} ({r.min_mean_curve_online/1000:.1f}) | "
                         f"{fmt(r.endpoint, r.endpoint_se)} | {r.endpoint_n}/{r.n} | {r.no_dip_observed_n}/{r.n} |")
        lines.append("")
    lines += ["¹ 학습 후 초기창의 실제 평가점 전체를 task별 순수 diffusion policy 평균과 비교. "
              "평가 사이의 성능을 보장하지 않으며, 평균곡선 dip 유무와 다른 지표다.", "",
              "후반 n이 작은 것은 대개 100k 예산 또는 다른 평가 간격 때문이다. 미완료 run이라는 뜻이 아니다. "
              "alr_half/double은 마지막 평가 126,896이라 엄격한 129,152 열에서 제외된다. "
              "별도 at125k_interpolated 열은 양쪽 관측이 있는 경우에만 선형 보간한다.", ""]
    (out / "ALL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def offline_analysis(logs, out):
    """Final training-batch diagnostics, not held-out calibration tests."""
    pattern = re.compile(r"^(td|cql|calql)_(can|square)_s([123])_log\.csv$")
    rows = []
    for path in sorted((logs / "pretrain").glob("*_log.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        method, task, seed = match.groups()
        df = pd.read_csv(path)
        critic = df[df.phase == method].drop_duplicates("step", keep="last").sort_values("step")
        distill = df[df.phase == "distill"].drop_duplicates("step", keep="last").sort_values("step")
        if critic.step.iloc[-1] != 50000 or distill.step.iloc[-1] != 25000:
            raise ValueError(f"Incomplete offline training log: {path}")
        tail = critic.tail(10)
        row = dict(task=task, method=method, seed=int(seed), critic_steps=50000,
                   distill_steps=25000, averaging_first_step=int(tail.step.iloc[0]),
                   averaging_last_step=int(tail.step.iloc[-1]),
                   source_file=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for key in ("q_mean", "q_ood_mean", "return_mean", "floored_frac", "penalty", "critic_loss"):
            row[key] = float(tail[key].mean())
        row["q_data_minus_G"] = row["q_mean"] - row["return_mean"]
        row["q_ood_minus_G"] = row["q_ood_mean"] - row["return_mean"]
        row["q_ood_minus_data"] = row["q_ood_mean"] - row["q_mean"]
        row["distill_loss_last10"] = float(distill.tail(10).noise_critic_loss.mean())
        rows.append(row)
    if not rows:
        return
    if len(rows) != 18:
        raise ValueError(f"Expected 18 latest TD/CQL/CalQL logs, got {len(rows)}")
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "offline_per_seed.csv", index=False)
    cols = ["q_mean", "q_ood_mean", "return_mean", "floored_frac", "q_data_minus_G",
            "q_ood_minus_G", "q_ood_minus_data", "distill_loss_last10"]
    summary = per_seed.groupby(["task", "method"])[cols].agg(["mean", "sem", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / "offline_groups.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected-runs", type=int, default=121)
    ap.add_argument("--previous-comparison", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    refs = reference_rows(args.logs)
    metrics, audits, evals, diagnostics, curves = [], [], [], [], {}
    for path in sorted(args.logs.iterdir()):
        match = PATTERN.match(path.name)
        if not path.is_dir() or not match:
            continue
        task, group, seed = match.groups()
        ref = float(refs.loc[refs.task == task, "mean"].iloc[0])
        m, a, ev, diag, yg = read_run(path, task, group, int(seed), ref)
        metrics.append(m); audits.append(a); evals.append(ev); diagnostics.append(diag)
        curves[path.name] = yg
    if len(metrics) != args.expected_runs:
        raise ValueError(f"Expected {args.expected_runs} scientific runs; read {len(metrics)}")
    per_seed = pd.DataFrame(metrics)
    if per_seed.duplicated(["task", "group", "seed"]).any():
        raise ValueError("Duplicate task/group/seed")
    groups = aggregate(per_seed, curves)
    ds = pd.concat(diagnostics, ignore_index=True)
    cols = [c for c in DIAGNOSTICS + ["entropy_estimate"] if c in ds]
    dg = ds.groupby(["task", "group", "bin_online"])[cols].agg(["mean", "sem", "count"])
    dg.columns = ["_".join(c) for c in dg.columns]
    tables = {"references": refs, "per_seed": per_seed, "groups": groups,
              "audit": pd.DataFrame(audits), "eval_clean": pd.concat(evals, ignore_index=True),
              "diagnostic_per_seed": ds, "diagnostic_groups": dg.reset_index(),
              "contrasts_seed_matched": contrasts(per_seed)}
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
    if args.previous_comparison:
        previous = pd.read_csv(args.previous_comparison)
        previous.loc[previous.task == "square", "group"] = previous.loc[previous.task == "square", "group"].str.removeprefix("square_")
        check = previous.merge(groups, on=["task", "group"], suffixes=("_old", "_new"), validate="one_to_one")
        check["auc_delta"] = check.auc_early - check.early_auc
        check["min_delta"] = check.min_mean_curve_new - check.min_mean_curve_old
        check["endpoint_delta"] = check.endpoint - check.endpoint_success
        check[["task", "group", "auc_delta", "min_delta", "endpoint_delta"]].to_csv(args.out / "previous_analysis_check.csv", index=False)
    write_tables(groups, args.out)
    offline_analysis(args.logs, args.out)
    print(groups[["task", "group", "n", "auc_early", "auc_early_se", "min_mean_curve", "endpoint", "endpoint_se", "endpoint_n"]].to_string(index=False))
    print("References:\n", refs.to_string(index=False))
    print(f"Read {len(per_seed)} runs in {len(groups)} groups. Outputs: {args.out.resolve()}")


if __name__ == "__main__":
    main()

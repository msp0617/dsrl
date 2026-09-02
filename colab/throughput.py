"""Report throughput and projected runtime from a run's train_log.csv.

    python colab/throughput.py <logdir> [--target 300000] [--warmup 2]

Rows are written by LoggingCallback and carry both the wall clock and the step
counter in original-environment steps, so the rate below is directly comparable
to the paper's x axis. The first rows are skipped because early wall time is
dominated by process start-up and the initial rollout.
"""

import argparse
import csv
import os
import time


def read_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        try:
            parsed.append(
                (
                    time.mktime(time.strptime(row["wall_time"], "%Y-%m-%d %H:%M:%S")),
                    int(row["env_steps"]),
                )
            )
        except (KeyError, ValueError):
            continue
    return parsed


def summarize(rows, target, warmup, act_steps=4):
    if len(rows) < warmup + 2:
        raise SystemExit(
            "need at least %d rows to measure, found %d" % (warmup + 2, len(rows))
        )
    rows = rows[warmup:]
    elapsed = rows[-1][0] - rows[0][0]
    steps = rows[-1][1] - rows[0][1]
    if elapsed <= 0 or steps <= 0:
        raise SystemExit("no progress between the first and last row")

    rate = steps / elapsed
    print("measured over   %d env steps in %.0f s (%d rows)" % (steps, elapsed, len(rows)))
    print("throughput      %.1f env steps/s" % rate)
    print("                %.2f chunk steps/s" % (rate / act_steps))
    remaining = max(target - rows[-1][1], 0)
    print("at %d env steps  %.1f h to go, %.1f h for the whole run"
          % (rows[-1][1], remaining / rate / 3600, target / rate / 3600))
    return rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="run directory holding train_log.csv")
    parser.add_argument("--target", type=int, default=300000, help="env steps in the full run")
    parser.add_argument("--warmup", type=int, default=2, help="leading rows to ignore")
    parser.add_argument("--act-steps", type=int, default=4, help="action chunk size")
    args = parser.parse_args()

    path = args.logdir
    if os.path.isdir(path):
        path = os.path.join(path, "train_log.csv")
    summarize(read_rows(path), args.target, args.warmup, args.act_steps)


if __name__ == "__main__":
    main()

"""Full inspection of every run and pre-training artifact in the Drive project folder.

Standard library only, so it runs on a CPU Colab runtime with just the Drive
mounted (no conda env), or locally on an unpacked csv_bundle:

    python scripts/inspect_runs.py --proj /content/drive/MyDrive/dsrl_project
    python scripts/inspect_runs.py --proj ~/Downloads --only can_td,can_cql,square_calql

Per run (``logs/<exp>/`` + ``logs/<exp>.out``): last write time (KST), the last
marker line of the .out ([done] / [eval] / Traceback / other), number of
evaluations and the last one, the resumable checkpoint (env steps per slot and
whether the model / replay-buffer / rng files are there; ``!`` marks a missing
file), and a one-word state:

    done      the .out ends with [done]
    error     the .out has a Traceback (the last error line is printed)
    stale     no [done], nothing written for --stale-min minutes: the VM died.
              Relaunching the same command resumes from the checkpoint shown,
              or from scratch when there is none.
    running   written recently

Then the pre-training artifacts (``logs/pretrain/*.pt``, ``logs/pretrain*.out``)
and the csv bundle. Finished runs are hidden unless --all or --only is given.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

KST = 9 * 3600
MARK = re.compile(r"\[(done|eval|resume|pretrain|budget)\]|Traceback|Error|Killed|MemoryError")


def kst(ts):
    return time.strftime("%m-%d %H:%M", time.gmtime(ts + KST)) if ts else "-"


def last_lines(path, n=400):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    return data.splitlines()[-n:]


def out_summary(path):
    """(last marker line, error line or None, ends with [done])"""
    lines = last_lines(path)
    marker, error = "", None
    for line in lines:
        if MARK.search(line):
            marker = line.strip()[:110]
        if line.startswith("Traceback") or "Error" in line or "Killed" in line:
            error = line.strip()[:160]
    if error and error.startswith("Traceback"):
        # the message itself is the last non-indented, non-empty line
        for line in reversed(lines):
            if line.strip() and not line.startswith(" "):
                error = line.strip()[:160]
                break
    done = any("[done]" in line for line in lines[-5:])
    return marker, error, done


def csv_tail(path):
    """(row count, last row) without pandas."""
    if not os.path.exists(path):
        return 0, None
    n, last = 0, None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n += 1
            last = row
    return n, last


def checkpoint_summary(ckpt_dir):
    state = os.path.join(ckpt_dir, "run_state.json")
    if not os.path.exists(state):
        return "none"
    try:
        raw = json.load(open(state))
    except (OSError, ValueError) as e:
        return "run_state.json unreadable (%s)" % e.__class__.__name__
    slots = raw.get("slots") or {raw.get("slot", "a"): raw}
    parts = []
    for slot in sorted(slots, key=lambda s: -int(slots[s].get("env_steps", -1))):
        env = int(slots[slot].get("env_steps", -1))
        files = []
        for name in ("model_%s.zip", "replay_buffer_%s.pkl", "rng_%s.pkl"):
            files.append(name[0] + ("" if os.path.exists(os.path.join(ckpt_dir, name % slot)) else "!"))
        parts.append("%s@%d[%s]" % (slot, env, "".join(files)))
    latest = raw.get("slot")
    return " ".join(parts) + ("  latest=%s" % latest if latest else "")


def latest_mtime(exp_dir, out_path):
    paths = [out_path] + glob.glob(os.path.join(exp_dir, "*.csv")) + glob.glob(os.path.join(exp_dir, "checkpoint", "*"))
    ts = [os.path.getmtime(p) for p in paths if os.path.exists(p)]
    return max(ts) if ts else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default="/content/drive/MyDrive/dsrl_project")
    ap.add_argument("--only", default="", help="comma-separated exp_id prefixes (default: every run)")
    ap.add_argument("--stale-min", type=float, default=20.0)
    ap.add_argument("--all", action="store_true", help="also list finished ([done]) runs")
    args = ap.parse_args()
    logs = os.path.join(os.path.expanduser(args.proj), "logs")
    if not os.path.isdir(logs):
        sys.exit("no logs dir at %s" % logs)
    prefixes = [p for p in args.only.split(",") if p]
    now = time.time()
    print("now %s KST   logs=%s" % (kst(now), logs))

    dirs = {os.path.basename(d) for d in glob.glob(os.path.join(logs, "*")) if os.path.isdir(d) and os.path.basename(d) != "pretrain"}
    outs = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(logs, "*.out")) if not os.path.basename(p).startswith("pretrain")}
    exps = sorted(dirs | outs)
    if prefixes:
        exps = [e for e in exps if any(e.startswith(p) for p in prefixes)]
    rows = []
    for exp in exps:
        exp_dir, out_path = os.path.join(logs, exp), os.path.join(logs, exp + ".out")
        marker, error, done = out_summary(out_path) if os.path.exists(out_path) else ("(no .out)", None, False)
        n_eval, last_eval = csv_tail(os.path.join(exp_dir, "eval_log.csv"))
        n_train, last_train = csv_tail(os.path.join(exp_dir, "train_log.csv"))
        mtime = latest_mtime(exp_dir, out_path)
        age_min = (now - mtime) / 60 if mtime else float("inf")
        if done:
            state = "done"
        elif error:
            state = "error"
        elif age_min > args.stale_min:
            state = "stale"
        else:
            state = "running"
        rows.append((mtime, exp, state, marker, error, n_eval, last_eval, last_train, checkpoint_summary(os.path.join(exp_dir, "checkpoint"))))

    rows.sort(key=lambda r: -r[0])
    shown = [r for r in rows if args.all or prefixes or r[2] != "done"]
    print("\n== runs (%d total, %d shown; newest write first; times KST) ==" % (len(rows), len(shown)))
    for mtime, exp, state, marker, error, n_eval, last_eval, last_train, ckpt in shown:
        ev = "no eval"
        if last_eval:
            ev = "%d evals, last env=%s succ=%.2f" % (n_eval, last_eval.get("env_steps"), float(last_eval.get("success_rate", "nan")))
        tr = "train env=%s" % last_train.get("env_steps") if last_train else "train -"
        print("%-26s %-7s write %s | %s | %s | ckpt %s" % (exp, state, kst(mtime), ev, tr, ckpt))
        print("    out: %s" % (marker or "-"))
        if error:
            print("    ERR: %s" % error)
    counts = {}
    for r in rows:
        counts[r[2]] = counts.get(r[2], 0) + 1
    print("summary:", ", ".join("%s %d" % kv for kv in sorted(counts.items())))

    pre = os.path.join(logs, "pretrain")
    pts = sorted(glob.glob(os.path.join(pre, "*.pt")), key=os.path.getmtime, reverse=True)
    print("\n== pretrain .pt (%d, newest first) ==" % len(pts))
    for p in pts:
        print("%-24s %6.1f MB  %s" % (os.path.basename(p), os.path.getsize(p) / 1e6, kst(os.path.getmtime(p))))
    pouts = sorted(glob.glob(os.path.join(logs, "pretrain*.out")), key=os.path.getmtime, reverse=True)
    print("\n== pretrain .out (%d) ==" % len(pouts))
    for p in pouts:
        marker, error, done = out_summary(p)
        print("%-30s %s  %s%s" % (os.path.basename(p), kst(os.path.getmtime(p)), "done " if done else "", (marker or "-")[:90]))
        if error:
            print("    ERR: %s" % error)
    bundle = os.path.join(os.path.expanduser(args.proj), "csv_bundle.zip")
    if os.path.exists(bundle):
        print("\ncsv_bundle.zip %.1f MB  %s" % (os.path.getsize(bundle) / 1e6, kst(os.path.getmtime(bundle))))


if __name__ == "__main__":
    main()

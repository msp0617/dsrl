"""Package this CPU analysis and text-only raw inputs for another computer.

No checkpoints, credentials, environments, or unrelated files are included.
Existing ZIPs are never overwritten. Run after analysis and figure generation.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--square-check", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = []
    for folder in ("logs/audit_20260908", "logs/analysis_source_20260908", "results/2026-09-08"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".png", ".pdf", ".txt"}:
                sources.append((path, path.relative_to(root).as_posix()))
    for name in ("analyze_completed_runs.py", "test_completed_analysis.py",
                 "plot_completed_analysis.py", "package_completed_analysis.py"):
        sources.append((root / "scripts" / name, "scripts/" + name))
    sources.append((root / "HANDOFF_2026-09-08.md", "HANDOFF_2026-09-08.md"))
    if args.square_check:
        sources.append((args.square_check.resolve(), "sources/square_offline_check_pasted.txt"))
    manifest = [{"path": name, "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path, name in sources]
    if len({m["path"] for m in manifest}) != len(manifest):
        raise ValueError("Duplicate archive member")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in sources:
            archive.write(path, name)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.out) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        for entry in manifest:
            data = archive.read(entry["path"])
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Archive hash mismatch: {entry['path']}")
    print(f"Verified {len(manifest)} files: {args.out.resolve()} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

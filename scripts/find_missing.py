"""Find circuits that didn't successfully complete in a previous tag.

Reads the per-tag phasepoly_best/summary.csv and the original tasklist, then
prints a comma-separated list of circuit stems whose status is NOT 'ok'
(i.e. PENDING / hard-timeout / failed / OUT_OF_MEMORY / never-launched).

Output is meant to be piped into `run_phasepoly_pipeline.sh --circuits "$(...)"`.

Examples:

    # 1) compute missing for a previous tag, dropping a known-bad circuit:
    MISSING=$(python3 scripts/find_missing.py \\
        --tag phasepoly_0504 --skip 'hwb8_113(best)')
    bash scripts/run_phasepoly_pipeline.sh \\
        --folders general,larger_circuits/adder,larger_circuits/hwb,larger_circuits/mcx \\
        --circuits "$MISSING" --hard-timeout 40000 \\
        --tag phasepoly_0504_retry

    # 2) inspect what would be resubmitted (no run):
    python3 scripts/find_missing.py --tag phasepoly_0504 --verbose

    # 3) override the tasklist (useful if original tasklist was incomplete):
    python3 scripts/find_missing.py --tag phasepoly_0504 \\
        --expected-from-folders /path/to/phasepoly_best_*/
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_manifest_ok_set(manifest_csv: Path, status_col: int = 1) -> set[str]:
    """Return {circuit_name : status=='ok'} stems."""
    ok: set[str] = set()
    with manifest_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                ok.add(row["circuit"])
    return ok


def _read_summary_ok_set(summary_csv: Path) -> set[str]:
    """PhasePoly summary.csv: 'ok' = is_timeout=='false'."""
    ok: set[str] = set()
    with summary_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("is_timeout") or "").lower() == "false":
                ok.add(row["circuit_name"])
    return ok


def _expected_from_tasklist(tasklist: Path) -> list[str]:
    stems: list[str] = []
    with tasklist.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stems.append(Path(line).stem)
    return stems


def _expected_from_folders(folders: list[Path]) -> list[str]:
    stems: list[str] = []
    seen: set[str] = set()
    for d in folders:
        if not d.exists():
            print(f"warn: folder not found: {d}", file=sys.stderr)
            continue
        for q in sorted(d.glob("*.qasm")):
            s = q.stem
            if s in seen:
                continue
            seen.add(s)
            stems.append(s)
    return stems


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find_missing",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tag", required=True,
                   help="Experiment tag (i.e. results/<tag>/ subdir).")
    p.add_argument("--results-dir", default=None, type=Path,
                   help=f"Default: {_PROJECT_ROOT / 'results'}")
    p.add_argument("--tasklist", default=None, type=Path,
                   help="Override the tasklist used as 'expected'. "
                        "Default: read job_info.txt's TASKLIST.")
    p.add_argument("--expected-from-folders", default=None,
                   help="Comma-separated folder paths; override 'expected' by "
                        "globbing *.qasm there. Use this when the original "
                        "tasklist was incomplete (e.g. some folders missed).")
    p.add_argument("--skip", default=None,
                   help="Comma-separated circuit stems to omit from the output.")
    p.add_argument("--verbose", action="store_true",
                   help="Print breakdown to stderr (count + per-bucket lists).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    results_dir = (args.results_dir or (_PROJECT_ROOT / "results")).resolve()
    tag_dir = results_dir / args.tag
    if not tag_dir.is_dir():
        print(f"error: not a directory: {tag_dir}", file=sys.stderr)
        return 2

    # 'expected' set
    expected: list[str]
    if args.expected_from_folders:
        folders = [Path(p.strip()).expanduser() for p in args.expected_from_folders.split(",") if p.strip()]
        expected = _expected_from_folders([f.resolve() for f in folders])
    elif args.tasklist:
        expected = _expected_from_tasklist(args.tasklist.resolve())
    else:
        # read TASKLIST= line from job_info.txt
        ji = tag_dir / "job_info.txt"
        tl = None
        if ji.exists():
            for line in ji.read_text().splitlines():
                if line.startswith("TASKLIST="):
                    tl = Path(line.split("=", 1)[1].strip())
                    break
        if tl is None:
            print(f"error: no --tasklist given and no TASKLIST= in {ji}", file=sys.stderr)
            return 2
        expected = _expected_from_tasklist(tl)

    if not expected:
        print("error: 'expected' set is empty", file=sys.stderr)
        return 2

    # 'ok' set — try manifest.csv first, fall back to summary.csv
    manifest = tag_dir / "manifest.csv"
    summary = tag_dir / "phasepoly_best" / "summary.csv"
    if manifest.exists():
        ok = _read_manifest_ok_set(manifest)
        src = "manifest.csv"
    elif summary.exists():
        ok = _read_summary_ok_set(summary)
        src = "phasepoly_best/summary.csv"
    else:
        print(f"warn: no manifest.csv or summary.csv under {tag_dir}; "
              "treating all expected as missing", file=sys.stderr)
        ok = set()
        src = "(none)"

    skip = set()
    if args.skip:
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    missing = [s for s in expected if s not in ok and s not in skip]

    if args.verbose:
        seen_in_manifest = set()
        if manifest.exists():
            with manifest.open() as f:
                for row in csv.DictReader(f):
                    seen_in_manifest.add(row["circuit"])
        elif summary.exists():
            with summary.open() as f:
                for row in csv.DictReader(f):
                    seen_in_manifest.add(row["circuit_name"])

        never_ran = [s for s in expected if s not in seen_in_manifest]
        non_ok    = [s for s in expected if s in seen_in_manifest and s not in ok]
        ok_count  = len([s for s in expected if s in ok])

        print(f"# tag        : {args.tag}", file=sys.stderr)
        print(f"# results    : {tag_dir}", file=sys.stderr)
        print(f"# source     : {src}", file=sys.stderr)
        print(f"# expected   : {len(expected)}", file=sys.stderr)
        print(f"# ok         : {ok_count}", file=sys.stderr)
        print(f"# never_ran  : {len(never_ran)} {never_ran[:5]}{'...' if len(never_ran)>5 else ''}", file=sys.stderr)
        print(f"# non_ok     : {len(non_ok)} {non_ok[:5]}{'...' if len(non_ok)>5 else ''}", file=sys.stderr)
        if skip:
            print(f"# skipped    : {sorted(skip)}", file=sys.stderr)
        print(f"# to-resubmit: {len(missing)}", file=sys.stderr)

    print(",".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())

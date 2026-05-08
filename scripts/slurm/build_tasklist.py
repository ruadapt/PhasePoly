"""Build a Slurm-array tasklist from benchmark folders / circuit lists.

Writes one absolute .qasm path per line to --output, then prints the line
count to stdout (so you can do `N=$(python3 build_tasklist.py ...)` and use
$N as the array size).

Mirrors the input-discovery semantics of run_experiment.py:
  --benchmark-folders general,larger_circuits/adder   # under <repo>/benchmarks/
  --input-dir PATH                                    # repeatable, absolute or relative
  --circuits a,b,c                                    # filter by stem
  --benchmark-list FILE                               # one circuit name per line
  --profiles-json profiles.json                       # only emit circuits listed there

Examples:

    # All circuits in benchmarks/general/ and benchmarks/larger_circuits/:
    python scripts/slurm/build_tasklist.py \\
        --benchmark-folders general,larger_circuits/adder,larger_circuits/hwb,larger_circuits/mcx \\
        --output ~/jobs/all_tasklist.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
for _p in (_PROJECT_ROOT, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _resolve(p: Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _benchmark_folder_path(folder: str) -> Path:
    """Resolve a folder argument. Absolute paths and explicit cwd-relative
    paths (starting with '/', '~', '.', or '..') are taken literally. Anything
    else is resolved under <repo>/benchmarks/ — including nested paths like
    'larger_circuits/adder', so the same arg form works for both flat
    ('general') and nested benchmark layouts."""
    s = folder.strip()
    is_explicit_path = (
        s.startswith("/")
        or s.startswith("~")
        or s in (".", "..")
        or s.startswith("./")
        or s.startswith("../")
    )
    if is_explicit_path:
        return _resolve(Path(s).expanduser())
    return (_PROJECT_ROOT / "benchmarks" / s).resolve()


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_benchmark_list(path: Path) -> list[str]:
    names: list[str] = []
    with path.open() as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                names.append(Path(line).stem)
    return names


def _profiles_circuits(path: Path) -> set[str]:
    with path.open() as f:
        data = json.load(f)
    out: set[str] = set()
    for k, v in data.get("circuits", {}).items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        out.add(k)
    return out


def _enumerate_qasm(input_dirs: list[Path]) -> dict[str, Path]:
    by_stem: dict[str, Path] = {}
    for d in input_dirs:
        if not d.exists():
            print(f"warn: input dir does not exist: {d}", file=sys.stderr)
            continue
        for q in sorted(d.glob("*.qasm")):
            by_stem.setdefault(q.stem, q.resolve())
    return by_stem


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_tasklist",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--benchmark-folders", default=None,
                   help="Comma-separated folders under benchmarks/, e.g. general,larger_circuits/adder.")
    p.add_argument("--input-dir", action="append", type=Path, default=None,
                   help="Repeatable directory to enumerate .qasm files. "
                        "Mutually compatible with --benchmark-folders (combined).")
    p.add_argument("--circuits", default=None,
                   help="Comma-separated circuit stems to keep. Skips others.")
    p.add_argument("--benchmark-list", type=Path, default=None,
                   help="Text file with one circuit stem per line (# comments OK).")
    p.add_argument("--profiles-json", type=Path, default=None,
                   help="Only include circuits listed under .circuits in this JSON.")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the tasklist (one absolute .qasm path per line).")
    p.add_argument("--quiet", action="store_true",
                   help="Skip the trailing count print (still writes --output).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    input_dirs: list[Path] = []
    for folder in _split_csv(args.benchmark_folders):
        input_dirs.append(_benchmark_folder_path(folder))
    if args.input_dir:
        input_dirs.extend(_resolve(d) for d in args.input_dir)
    if not input_dirs:
        print("error: need --benchmark-folders or --input-dir", file=sys.stderr)
        return 2

    by_stem = _enumerate_qasm(input_dirs)
    if not by_stem:
        print("error: no .qasm files found under given dirs", file=sys.stderr)
        return 2

    keep_filter: set[str] | None = None
    if args.circuits:
        keep_filter = set(_split_csv(args.circuits))
    if args.benchmark_list:
        list_names = set(_read_benchmark_list(_resolve(args.benchmark_list)))
        keep_filter = (keep_filter & list_names) if keep_filter else list_names
    if args.profiles_json:
        prof_names = _profiles_circuits(_resolve(args.profiles_json))
        keep_filter = (keep_filter & prof_names) if keep_filter else prof_names

    selected_stems = sorted(by_stem.keys() if keep_filter is None
                            else (s for s in by_stem if s in keep_filter))
    if keep_filter is not None:
        missing = sorted(keep_filter - set(by_stem.keys()))
        if missing:
            print(f"warn: no .qasm match for: {', '.join(missing)}", file=sys.stderr)

    if not selected_stems:
        print("error: nothing to write to tasklist (filter eliminated everything)",
              file=sys.stderr)
        return 2

    out_path = _resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for stem in selected_stems:
            f.write(f"{by_stem[stem]}\n")

    if not args.quiet:
        print(len(selected_stems))
    return 0


if __name__ == "__main__":
    sys.exit(main())

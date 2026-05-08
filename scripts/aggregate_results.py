"""Rebuild <tag>/phasepoly_best/ from scratch by walking every circuit subdir.

Use this when you want to regenerate phasepoly_best/ on an old run, or after
manually editing/removing circuit subdirs. Normal experiments don't need
this — run_experiment.py updates phasepoly_best/ incrementally per circuit.

Usage:
    python scripts/aggregate_results.py --tag exp_2026_05_01_baseline
    python scripts/aggregate_results.py --tag-dir results/exp_2026_05_01_baseline --model phasepoly
"""
import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _phasepoly_best import phasepoly_best_dirname, rebuild_phasepoly_best


def _build_arg_parser():
    p = argparse.ArgumentParser(prog="aggregate_results", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tag", help="Look under ./results/<tag>/")
    g.add_argument("--tag-dir", type=Path, help="Path to the tag directory directly")
    p.add_argument("--results-dir", type=Path, default=Path("./results/"),
                   help="Top-level results directory (used with --tag). Default: ./results/")
    p.add_argument("--model", default="phasepoly",
                   help="Model label written into summary.csv. Default: phasepoly")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    tag_dir = args.tag_dir or (args.results_dir / args.tag)
    n = rebuild_phasepoly_best(tag_dir, model=args.model)
    print(f"rebuilt {phasepoly_best_dirname(tag_dir)}/ for {n} circuit(s) at {tag_dir / phasepoly_best_dirname(tag_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CLI for circuit equivalence verification (Qiskit + mqt.qcec).

================================================================================
HOW TO USE
================================================================================

Always run from the repo root (`ppho_circuit_optimization_isca-ae/`):

    cd ppho_circuit_optimization_isca-ae

There are three subcommands: `single`, `folder`, and `examples`.

--------------------------------------------------------------------------------
1) Verify ONE circuit pair
--------------------------------------------------------------------------------

    python -m scripts.verify_circuits single \\
        --original  PATH/to/original.qasm \\
        --compared  PATH/to/compared.qasm \\
        [--methods qiskit,qcec]   # default: both
        [--timeout  300]          # per-verification seconds (default 300 = 5 min)

Example:

    python -m scripts.verify_circuits single \\
        --original 'benchmarks/general/barenco_tof_4.qasm' \\
        --compared 'main_results/phasepoly_results/pps_best/barenco_tof_4(7).qasm'

--------------------------------------------------------------------------------
2) Verify TWO FOLDERS (every key once)
--------------------------------------------------------------------------------

    python -m scripts.verify_circuits folder \\
        --original-dir DIR \\
        --compared-dir DIR \\
        [--keys adder_8,tof_5,barenco_tof_5,...]   # comma list
        [--keys-file PATH]                         # one key per line; '#' = comment
        [--methods qiskit,qcec]                    # default: both
        [--timeout 300]                            # default 300 (5 min)
        [--log     PATH]                           # mirror stdout/stderr to file

If neither --keys nor --keys-file is supplied, the built-in
`DEFAULT_CIRCUIT_KEYS` list is used (see src/circuit_verification.py).

File-to-key matching uses the "longest matching key wins" rule, so
`barenco_tof_5(7).qasm` binds to `barenco_tof_5` (NOT `tof_5`), and
`tof_5(7).qasm` binds to `tof_5`. Files like `adder_8_optimized.qasm`,
`adder_8(optimized).qasm`, and `optimized_adder_8.qasm` all bind to `adder_8`.

--------------------------------------------------------------------------------
3) Run the BUILT-IN demos
--------------------------------------------------------------------------------

    python -m scripts.verify_circuits examples [--timeout 300]

This runs both a single-pair example and a folder-vs-folder example, with:
    original = benchmarks/general/                       (reference circuits)
    compared = main_results/phasepoly_results/pps_best/  (optimized PPS outputs)

The folder-example log is written to
`results/verification/verify_circuits_examples_<TS>.log`.

--------------------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------------------

For each method, the result is `<status> <detail> (<elapsed>s)` where status is:

    ok       - verification finished; `detail` is the result string.
    skipped  - precondition failed (e.g. Qiskit is gated to <= 8 qubits).
    timeout  - the worker process was killed after `--timeout` seconds.
    error    - the worker raised an exception; `detail` is its message.

The `--timeout` is enforced via a child process (`multiprocessing` spawn);
this is the only reliable way to kill `mqt.qcec.verify` (a C++ extension that
ignores Python signals) on circuits like `mod_adder_1024`.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.circuit_verification import (  # noqa: E402
    ALL_METHODS,
    DEFAULT_CIRCUIT_KEYS,
    DEFAULT_TIMEOUT,
    verify_folder_pair,
    verify_pair,
)


def _parse_methods(s: str) -> list[str]:
    methods = [m.strip() for m in s.split(",") if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown method(s): {bad}; choose from {list(ALL_METHODS)}"
        )
    return methods


def _parse_keys(s: str) -> list[str]:
    return [k.strip() for k in s.split(",") if k.strip()]


def _load_keys_file(path: str) -> list[str]:
    keys: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                keys.append(line)
    return keys


def _print_single_record(record: dict) -> None:
    print(f"original: {record['original']}")
    print(f"compared: {record['compared']}")
    for method in ALL_METHODS:
        if method in record:
            m = record[method]
            print(f"  {method:>6}: {m['status']:<8} {m['detail']}  ({m['elapsed']:.2f}s)")


def cmd_single(args: argparse.Namespace) -> int:
    record = verify_pair(
        args.original, args.compared, methods=args.methods, timeout=args.timeout
    )
    _print_single_record(record)
    return 0


def cmd_folder(args: argparse.Namespace) -> int:
    if args.keys and args.keys_file:
        print("error: --keys and --keys-file are mutually exclusive", file=sys.stderr)
        return 2
    if args.keys:
        keys = args.keys
    elif args.keys_file:
        keys = _load_keys_file(args.keys_file)
    else:
        keys = DEFAULT_CIRCUIT_KEYS
    verify_folder_pair(
        keys=keys,
        original_folder=args.original_dir,
        compared_folder=args.compared_dir,
        methods=args.methods,
        timeout=args.timeout,
        log_path=args.log,
    )
    return 0


def cmd_examples(args: argparse.Namespace) -> int:
    # Examples verify the optimized PPS results against the reference benchmarks:
    #   original = benchmarks/general/                       (reference circuits)
    #   compared = main_results/phasepoly_results/pps_best/  (optimized outputs)
    bench_dir = REPO_ROOT / "benchmarks" / "general"
    pps_dir = REPO_ROOT / "main_results" / "phasepoly_results" / "pps_best"

    print("=" * 72)
    print("Example 1: single pair (barenco_tof_4: benchmarks/general vs pps_best)")
    print("=" * 72)
    # barenco_tof_4 has 7 qubits, so BOTH qiskit (<=8 qubits gate) and qcec exercise.
    single_orig = bench_dir / "barenco_tof_4.qasm"
    single_comp = pps_dir / "barenco_tof_4(7).qasm"
    if not single_orig.is_file() or not single_comp.is_file():
        print(f"  [skip] missing file(s):\n    {single_orig}\n    {single_comp}")
    else:
        record = verify_pair(
            str(single_orig), str(single_comp),
            methods=args.methods, timeout=args.timeout,
        )
        _print_single_record(record)

    print()
    print("=" * 72)
    print(f"Example 2: folder vs folder  ({bench_dir.name}  vs  {pps_dir.name})")
    print("=" * 72)
    if not bench_dir.is_dir() or not pps_dir.is_dir():
        print(f"  [skip] missing folder(s):\n    {bench_dir}\n    {pps_dir}")
    else:
        log_dir = REPO_ROOT / "results" / "verification"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"verify_circuits_examples_{timestamp}.log"
        verify_folder_pair(
            keys=DEFAULT_CIRCUIT_KEYS,
            original_folder=str(bench_dir),
            compared_folder=str(pps_dir),
            methods=args.methods,
            timeout=args.timeout,
            log_path=str(log_path),
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Circuit equivalence verification (Qiskit + mqt.qcec)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--methods", type=_parse_methods, default=list(ALL_METHODS),
        help=f"comma-separated subset of {list(ALL_METHODS)} (default: both)",
    )
    common.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"per-verification timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )

    p_single = sub.add_parser("single", parents=[common], help="verify one circuit pair")
    p_single.add_argument("--original", required=True, help="path to original .qasm")
    p_single.add_argument("--compared", required=True, help="path to compared .qasm")
    p_single.set_defaults(func=cmd_single)

    p_folder = sub.add_parser("folder", parents=[common], help="verify two folders")
    p_folder.add_argument("--original-dir", required=True)
    p_folder.add_argument("--compared-dir", required=True)
    p_folder.add_argument(
        "--keys", type=_parse_keys, default=None,
        help="comma-separated circuit keys (default: built-in DEFAULT_CIRCUIT_KEYS)",
    )
    p_folder.add_argument(
        "--keys-file", default=None,
        help="file with one circuit key per line (mutually exclusive with --keys)",
    )
    p_folder.add_argument("--log", default=None, help="optional log file path")
    p_folder.set_defaults(func=cmd_folder)

    p_examples = sub.add_parser(
        "examples", parents=[common],
        help="run the built-in single + folder demonstration",
    )
    p_examples.set_defaults(func=cmd_examples)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

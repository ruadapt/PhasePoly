"""Run PhasePoly on a SINGLE circuit (no pool, no folder enumeration).

This is the unit of work for Slurm job arrays. Spawns _run_one_circuit.py
under a wall-clock timeout, recovers (best).qasm if the worker is killed,
then upserts the per-tag phasepoly_best/summary.csv row.

CLI is parametric — pass either --rounds-json directly, or --profiles-json
plus the circuit name (rounds are looked up from the profiles file). When
neither is given, the in-file ROUNDS default from run_experiment.py is used.

Examples:

    # rounds inlined:
    python scripts/run_phasepoly_one.py \\
        --circuit-name tof_3 \\
        --input-qasm benchmarks/general/tof_3.qasm \\
        --output-dir results/my_tag/tof_3 \\
        --tag my_tag \\
        --log-path results/my_tag/log.txt \\
        --rounds-json '[{"method":"row_heap","rotation_merging_mode":"advanced_rotation_merging","heap_size":1000,"ends_checked":1000,"group_size":1}]' \\
        --timeout 600 --hard-timeout 600

    # rounds from super_parameters.json (per-circuit profile assignments):
    python scripts/run_phasepoly_one.py \\
        --circuit-name adder_8 \\
        --input-qasm benchmarks/general/adder_8.qasm \\
        --output-dir results/big_run/adder_8 \\
        --tag big_run \\
        --log-path results/big_run/log.txt \\
        --profiles-json benchmarks/scripts/config/super_parameters.json \\
        --timeout 7200 --hard-timeout 9000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _phasepoly_best import update_phasepoly_best
from run_experiment import (
    ROUNDS,
    TIMEOUT_SECONDS,
    WORKER_SCRIPT,
    _append_log,
    _load_profiles_json,
    _maybe_write_best_after_timeout,
    _validate_rounds,
)


def _resolve_rounds(args: argparse.Namespace) -> list[dict]:
    if args.rounds_json:
        rounds = json.loads(args.rounds_json)
    elif args.profiles_json:
        profiles_rounds = _load_profiles_json(args.profiles_json)
        if args.circuit_name not in profiles_rounds:
            print(
                f"warn: circuit '{args.circuit_name}' not in profiles "
                f"{args.profiles_json}; falling back to default ROUNDS",
                file=sys.stderr,
            )
            rounds = ROUNDS
        else:
            rounds = profiles_rounds[args.circuit_name]
    else:
        rounds = ROUNDS

    if args.gaussian_elim_algorithm is not None:
        for r in rounds:
            r.setdefault("gaussian_elim_algorithm", args.gaussian_elim_algorithm)

    if not isinstance(rounds, list) or not rounds:
        raise ValueError("resolved rounds list is empty")
    _validate_rounds(rounds)
    return rounds


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_phasepoly_one",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--circuit-name", required=True)
    p.add_argument("--input-qasm", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Per-circuit output dir, normally <results>/<tag>/<circuit>")
    p.add_argument("--tag", required=True)
    p.add_argument("--log-path", required=True, type=Path,
                   help="Shared log file for the tag (multi-process safe).")
    rounds_group = p.add_mutually_exclusive_group()
    rounds_group.add_argument("--rounds-json", default=None,
                              help="JSON array of round-parameter dicts. "
                                   "Mutually exclusive with --profiles-json.")
    rounds_group.add_argument("--profiles-json", type=Path, default=None,
                              help="Legacy super_parameters.json; rounds derived for "
                                   "--circuit-name. Falls back to default ROUNDS if "
                                   "the circuit is not listed.")
    p.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS,
                   help=f"Soft per-circuit budget recorded in the log. Default: {TIMEOUT_SECONDS}")
    p.add_argument("--hard-timeout", type=int, default=None,
                   help="Wall-clock kill enforced here. Default: --timeout")
    p.add_argument("--model", default="phasepoly",
                   help="Model label written into phasepoly_best/summary.csv. Default: phasepoly")
    p.add_argument("--gaussian-elim-algorithm", choices=["modified", "classic"], default=None,
                   help="Stamp gaussian_elim_algorithm onto every round that doesn't set it.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    hard_timeout = args.hard_timeout if args.hard_timeout is not None else args.timeout
    if hard_timeout <= 0:
        print("error: --hard-timeout must be positive", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("error: --timeout must be positive", file=sys.stderr)
        return 2

    input_path = args.input_qasm if args.input_qasm.is_absolute() else (
        Path.cwd() / args.input_qasm
    ).resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (
        Path.cwd() / args.output_dir
    ).resolve()
    log_path = args.log_path if args.log_path.is_absolute() else (
        Path.cwd() / args.log_path
    ).resolve()

    if not input_path.exists():
        print(f"error: input qasm not found: {input_path}", file=sys.stderr)
        return 2

    rounds = _resolve_rounds(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(WORKER_SCRIPT),
        "--circuit-name", args.circuit_name,
        "--input-qasm", str(input_path),
        "--output-dir", str(output_dir),
        "--tag", args.tag,
        "--log-path", str(log_path),
        "--rounds-json", json.dumps(rounds),
    ]

    _append_log(
        log_path,
        f"{args.circuit_name} START rounds={len(rounds)} "
        f"timeout={args.timeout}s hard_timeout={hard_timeout}s",
    )
    t0 = time.monotonic()
    is_timeout = False
    rc: int | str = 0
    try:
        proc = subprocess.run(cmd, timeout=hard_timeout, check=False,
                              cwd=str(_PROJECT_ROOT))
        elapsed = time.monotonic() - t0
        rc = proc.returncode
        tag_str = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        _append_log(log_path, f"{args.circuit_name} END {tag_str} elapsed={elapsed:.1f}s")
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        is_timeout = True
        rc = "TIMEOUT"
        _append_log(
            log_path,
            f"{args.circuit_name} TIMEOUT after {elapsed:.1f}s (hard={hard_timeout}s)",
        )
        _maybe_write_best_after_timeout(output_dir, args.circuit_name, input_path)
    except Exception as e:  # noqa: BLE001
        rc = f"EXCEPTION:{e}"
        _append_log(log_path, f"{args.circuit_name} HOST_EXCEPTION: {e}")
        _maybe_write_best_after_timeout(output_dir, args.circuit_name, input_path)

    try:
        update_phasepoly_best(output_dir.parent, args.circuit_name,
                              is_timeout=is_timeout, model=args.model)
    except Exception as e:  # noqa: BLE001
        _append_log(log_path, f"{args.circuit_name} PHASEPOLY_BEST_UPDATE_ERROR: {e}")

    if isinstance(rc, int):
        return rc
    return 0 if is_timeout else 1


if __name__ == "__main__":
    sys.exit(main())

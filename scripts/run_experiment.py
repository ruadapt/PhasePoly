"""Main experiment driver: run phasepoly synthesis on a list of circuits,
each with multiple chained rounds, with per-circuit total timeout.

The experiment configuration lives at the top of this file as plain Python
constants. The CLI lets you override the most common knobs without editing
the file (useful for splitting work across tmux sessions under a shared tag).

Usage examples:

    # Use the in-file config:
    python scripts/run_experiment.py

    # Run only some circuits (multi-tmux split):
    python scripts/run_experiment.py --tag shared_run --circuits adder_8,grover_5

    # Override the input dir and timeout:
    python scripts/run_experiment.py --input-dir ./benchmarks/general/ --timeout 1200

    # Inline a custom rounds schedule:
    python scripts/run_experiment.py --rounds-json '[{"method":"single_block_greedy"}]'

    # Use the legacy super_parameters.json (profiles + per-circuit assign):
    python scripts/run_experiment.py --profiles-json evaluation/super_parameters.json
"""
import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.utils import is_in_testing_circuit_names
from _phasepoly_best import update_phasepoly_best

# ============================================================================
# EXPERIMENT CONFIG (edit these defaults; CLI flags override at runtime)
# ============================================================================

TAG = "exp_default"

INPUT_DIR = Path("./benchmarks/general/")
RESULTS_DIR = Path("./results/")
TIMEOUT_SECONDS = 600          # per-circuit total wall-clock budget

# Each entry = one round. Round N+1 reads round N's output (chained).
# Allowed keys: method, rotation_merging_mode, heap_size, ends_checked, group_size.
# Default: 7 rounds, heap_size=ends_checked=1000 throughout, alternating
# group_size 1 / 3 / 1 / 5 / 1 / 7 / 1.
_BASE = {
    "method": "row_heap",
    "rotation_merging_mode": "advanced_rotation_merging",
    "heap_size": 1000,
    "ends_checked": 1000,
}
ROUNDS = [
    {**_BASE, "group_size": 1},
    {**_BASE, "group_size": 3},
    {**_BASE, "group_size": 1},
    {**_BASE, "group_size": 5},
    {**_BASE, "group_size": 1},
    {**_BASE, "group_size": 7},
    {**_BASE, "group_size": 1},
]

# ============================================================================

WORKER_SCRIPT = _PROJECT_ROOT / "scripts" / "_run_one_circuit.py"
ALLOWED_ROUND_KEYS = {"method", "rotation_merging_mode", "heap_size",
                      "ends_checked", "group_size", "gaussian_elim_algorithm"}


def _validate_rounds(rounds):
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("ROUNDS must be a non-empty list")
    for i, r in enumerate(rounds, 1):
        bad = set(r) - ALLOWED_ROUND_KEYS
        if bad:
            raise ValueError(f"Round {i} has unknown keys: {bad}; allowed: {ALLOWED_ROUND_KEYS}")


def _append_log(log_path: Path, message: str) -> None:
    """Append-only writes; safe across tmux sessions (POSIX O_APPEND atomicity)."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def _load_profiles_json(path: Path) -> dict:
    """Load legacy super_parameters.json. Returns dict mapping circuit_name -> rounds list.
    Keys starting with '_' (e.g. '_comment', '_section_general') are treated as
    inline comments and skipped."""
    with open(path) as f:
        data = json.load(f)
    profiles = {k: v for k, v in data.get("profiles", {}).items()
                if not k.startswith("_") and isinstance(v, dict)}
    circuits_cfg = data.get("circuits", {})
    out = {}
    for circuit_name, cfg in circuits_cfg.items():
        if circuit_name.startswith("_") or not isinstance(cfg, dict):
            continue
        rounds = []
        for profile_name in cfg.get("assign", []):
            if profile_name not in profiles:
                print(f"warn: profile '{profile_name}' missing for {circuit_name}; "
                      "skipping that round", file=sys.stderr)
                continue
            base = profiles[profile_name].copy()
            # Legacy profiles only define heap_size/ends_checked/group_size;
            # default the rest to row_heap + advanced merging.
            base.setdefault("method", "row_heap")
            base.setdefault("rotation_merging_mode", "advanced_rotation_merging")
            rounds.append(base)
        if rounds:
            out[circuit_name] = rounds
    return out


def _maybe_write_best_after_timeout(output_dir: Path, circuit_name: str,
                                    input_qasm: Path) -> None:
    """If the worker was killed before writing (best), try to recover.
    Prefer the last successful round; fall back to copying the original
    input so every circuit always has a (best).qasm."""
    progress_path = output_dir / "_progress.json"
    best_qasm = output_dir / f"{circuit_name}(best).qasm"
    if best_qasm.exists():
        return
    progress = {}
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                progress = json.load(f)
        except Exception:
            progress = {}
    last = progress.get("last_successful_round", 0) if progress else 0
    if last:
        src = output_dir / f"{circuit_name}({last}).qasm"
        if src.exists():
            shutil.copyfile(src, best_qasm)
            body = {
                "best_round": last,
                "circuit_name": circuit_name,
                "tag": progress.get("tag"),
                "round_data": progress["rounds"].get(str(last)),
                "note": "written by main after worker timeout/crash",
            }
            with open(output_dir / f"{circuit_name}(best).txt", "w") as f:
                f.write(f"# (best) = round {last}  [recovered after timeout]\n")
                f.write(json.dumps(body, indent=2, default=str))
                f.write("\n")
            return
    # Full fallback: no successful round at all — copy the original input.
    if input_qasm.exists():
        shutil.copyfile(input_qasm, best_qasm)
        body = {
            "best_round": 0,
            "circuit_name": circuit_name,
            "tag": progress.get("tag") if progress else None,
            "round_data": None,
            "note": "no round completed (timeout / crash); (best) is the original input",
        }
        with open(output_dir / f"{circuit_name}(best).txt", "w") as f:
            f.write("# (best) = original input  [no round completed]\n")
            f.write(json.dumps(body, indent=2, default=str))
            f.write("\n")


def _build_arg_parser():
    p = argparse.ArgumentParser(prog="run_experiment", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default=None,
                   help=f"Experiment tag (output goes to RESULTS_DIR/<tag>/). Default: {TAG}")
    p.add_argument("--circuits", default=None,
                   help="Comma-separated circuit name list. If omitted, runs all "
                        "QASM files in --input-dir that match a profiles entry "
                        "(or all QASM files when no profiles JSON is given).")
    p.add_argument("--input-dir", default=None, type=Path,
                   help=f"Directory containing input .qasm files. Default: {INPUT_DIR}")
    p.add_argument("--results-dir", default=None, type=Path,
                   help=f"Top-level results directory. Default: {RESULTS_DIR}")
    p.add_argument("--timeout", type=int, default=None,
                   help=f"Per-circuit total wall-clock timeout (seconds). Default: {TIMEOUT_SECONDS}")
    rounds_source = p.add_mutually_exclusive_group()
    rounds_source.add_argument("--profiles-json", type=Path, default=None,
                               help="Use legacy super_parameters.json (profiles + assign). "
                                    "When set, overrides ROUNDS for every circuit listed in the JSON.")
    rounds_source.add_argument("--rounds-json", default=None,
                               help="JSON-encoded list of round-parameter dicts. "
                                    "Overrides the in-file ROUNDS for every selected circuit.")
    p.add_argument("--model", default="phasepoly",
                   help="Model label written into phasepoly_best/summary.csv. Default: phasepoly")
    p.add_argument("--gaussian-elim-algorithm", choices=["modified", "classic"], default=None,
                   help="Override the Gaussian-elimination backend for every round in this run. "
                        "When set, injects 'gaussian_elim_algorithm' into each round dict that "
                        "does not already specify one. Default: leave each round unchanged "
                        "(phasepoly_synthesize falls back to 'modified').")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned commands without running anything.")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    tag = args.tag or TAG
    # Resolve to absolute paths so the worker subprocess (run with cwd=project_root)
    # and the main process see the same files regardless of where the user invoked us.
    input_dir = (args.input_dir or INPUT_DIR)
    results_dir = (args.results_dir or RESULTS_DIR)
    if not input_dir.is_absolute():
        input_dir = (Path.cwd() / input_dir).resolve()
    if not results_dir.is_absolute():
        results_dir = (Path.cwd() / results_dir).resolve()
    timeout = args.timeout if args.timeout is not None else TIMEOUT_SECONDS

    if args.rounds_json:
        try:
            base_rounds = json.loads(args.rounds_json)
        except json.JSONDecodeError as e:
            print(f"error: invalid --rounds-json: {e}", file=sys.stderr)
            return 2
    else:
        base_rounds = ROUNDS

    try:
        _validate_rounds(base_rounds)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Resolve which rounds-config to use for each circuit.
    profiles_rounds = _load_profiles_json(args.profiles_json) if args.profiles_json else None

    # Optional global override: stamp `gaussian_elim_algorithm` onto every round
    # that does not already declare one. Applied to both the default ROUNDS and
    # the per-circuit profile rounds so the rest of the pipeline does not have
    # to know about the flag.
    if args.gaussian_elim_algorithm is not None:
        algo = args.gaussian_elim_algorithm
        for r in base_rounds:
            r.setdefault("gaussian_elim_algorithm", algo)
        if profiles_rounds is not None:
            for rounds in profiles_rounds.values():
                for r in rounds:
                    r.setdefault("gaussian_elim_algorithm", algo)

    # Build the circuit -> qasm-filename map.
    if not input_dir.exists():
        print(f"error: input dir does not exist: {input_dir}", file=sys.stderr)
        return 2
    qasm_files = [f for f in os.listdir(input_dir) if f.endswith(".qasm")]

    if args.circuits:
        wanted = [c.strip() for c in args.circuits.split(",") if c.strip()]
        qasm_files_pair = is_in_testing_circuit_names(qasm_files, wanted)
        # Drop user-supplied names that don't have a matching file.
        missing = [c for c in wanted if c not in qasm_files_pair]
        if missing:
            print(f"warn: no .qasm file found for: {missing}", file=sys.stderr)
    elif profiles_rounds:
        qasm_files_pair = is_in_testing_circuit_names(qasm_files, list(profiles_rounds.keys()))
    else:
        # No filter: derive circuit names from filenames (strip .qasm).
        qasm_files_pair = {Path(f).stem: f for f in qasm_files}

    if not qasm_files_pair:
        print("error: no input circuits selected; check --circuits or --input-dir",
              file=sys.stderr)
        return 2

    # Tag directory + log + per-session config.
    tag_dir = results_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    log_path = tag_dir / "log.txt"

    session_id = f"{int(time.time())}_{os.getpid()}"
    session_config = {
        "session_id": session_id,
        "tag": tag,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "results_dir": str(results_dir),
        "timeout_seconds": timeout,
        "circuits": sorted(qasm_files_pair.keys()),
        "rounds_default": base_rounds,
        "rounds_json": args.rounds_json,
        "profiles_json": str(args.profiles_json) if args.profiles_json else None,
        "gaussian_elim_algorithm": args.gaussian_elim_algorithm,
    }
    with open(tag_dir / f"config_{session_id}.json", "w") as f:
        json.dump(session_config, f, indent=2, default=str)

    _append_log(log_path, f"=== session {session_id} START tag={tag} "
                          f"circuits={len(qasm_files_pair)} timeout={timeout}s ===")

    # Main loop: one circuit at a time, each via subprocess with timeout.
    exit_codes = {}
    for circuit_name in sorted(qasm_files_pair.keys()):
        qasm_filename = qasm_files_pair[circuit_name]
        input_path = input_dir / qasm_filename
        output_dir = tag_dir / circuit_name

        rounds_for_this = profiles_rounds.get(circuit_name, base_rounds) if profiles_rounds else base_rounds

        cmd = [
            sys.executable, str(WORKER_SCRIPT),
            "--circuit-name", circuit_name,
            "--input-qasm", str(input_path),
            "--output-dir", str(output_dir),
            "--tag", tag,
            "--log-path", str(log_path),
            "--rounds-json", json.dumps(rounds_for_this),
        ]
        if args.dry_run:
            print(" ".join(map(str, cmd)))
            continue

        _append_log(log_path, f"{circuit_name} START rounds={len(rounds_for_this)} "
                              f"timeout={timeout}s")
        t0 = time.monotonic()
        is_timeout = False
        try:
            proc = subprocess.run(cmd, timeout=timeout, check=False,
                                  cwd=str(_PROJECT_ROOT))
            elapsed = time.monotonic() - t0
            exit_codes[circuit_name] = proc.returncode
            tag_str = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
            _append_log(log_path, f"{circuit_name} END {tag_str} elapsed={elapsed:.1f}s")
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            is_timeout = True
            exit_codes[circuit_name] = "TIMEOUT"
            _append_log(log_path, f"{circuit_name} TIMEOUT after {elapsed:.1f}s")
            _maybe_write_best_after_timeout(output_dir, circuit_name, input_path)
        except Exception as e:
            exit_codes[circuit_name] = f"EXCEPTION:{e}"
            _append_log(log_path, f"{circuit_name} HOST_EXCEPTION: {e}")
            _maybe_write_best_after_timeout(output_dir, circuit_name, input_path)

        # Per-circuit phasepoly_best update: copy (best) artifacts + upsert CSV row.
        # Multi-tmux sessions share the CSV via fcntl.flock.
        try:
            update_phasepoly_best(tag_dir, circuit_name, is_timeout=is_timeout, model=args.model)
        except Exception as e:
            _append_log(log_path, f"{circuit_name} PHASEPOLY_BEST_UPDATE_ERROR: {e}")

    _append_log(log_path, f"=== session {session_id} END results: {exit_codes} ===")
    print(f"session {session_id} done. results dir: {tag_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

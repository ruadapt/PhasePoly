"""Worker script: run all rounds of a single circuit, chained.

Invoked by scripts/run_experiment.py as a subprocess so a per-circuit
timeout can be enforced. Each round's output feeds the next round's input.
After the last successful round, writes {circuit}(best).qasm + .txt.

CLI:
    python scripts/_run_one_circuit.py \
        --circuit-name X \
        --input-qasm Y \
        --output-dir Z \
        --tag T \
        --log-path L \
        --rounds-json '[{...}, {...}]'
"""
import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# Ensure the project root is on sys.path so `import src.phasepoly` works
# regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.phasepoly import (
    phasepoly_synthesize,
    PhasepolySynthesisError,
    PhasepolySynthesisResult,
)

def _append_log(log_path: Path, message: str) -> None:
    """Append a single line to the shared log. POSIX O_APPEND makes small
    writes atomic, so multiple sessions can share log_path safely."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically by writing to .tmp then renaming."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _dump_round_txt(path: Path, round_index: int, result: PhasepolySynthesisResult) -> None:
    """Write the human-readable .txt sidecar for one round."""
    body = {
        "round": round_index,
        **dataclasses.asdict(result),
    }
    with open(path, "w") as f:
        f.write(f"# Round {round_index} of circuit '{result.circuit_name}'\n")
        f.write(json.dumps(body, indent=2, default=str))
        f.write("\n")


def _dump_best_txt(path: Path, best_round: int, progress: dict) -> None:
    body = {
        "best_round": best_round,
        "circuit_name": progress.get("circuit_name"),
        "tag": progress.get("tag"),
        "round_data": progress["rounds"].get(str(best_round)),
    }
    with open(path, "w") as f:
        f.write(f"# (best) = round {best_round}\n")
        f.write(json.dumps(body, indent=2, default=str))
        f.write("\n")


def _maybe_write_best(output_dir: Path, circuit_name: str, progress: dict,
                      input_qasm: Path) -> None:
    """Copy the last successful round to (best).qasm + .txt. If no round
    succeeded, fall back to copying the original input so every circuit
    always has a (best).qasm. Idempotent."""
    last = progress.get("last_successful_round", 0)
    if last:
        src = output_dir / f"{circuit_name}({last}).qasm"
        if src.exists():
            shutil.copyfile(src, output_dir / f"{circuit_name}(best).qasm")
            _dump_best_txt(output_dir / f"{circuit_name}(best).txt", last, progress)
            return
    # Fallback: no successful round — preserve the original input as best.
    if input_qasm.exists():
        shutil.copyfile(input_qasm, output_dir / f"{circuit_name}(best).qasm")
        body = {
            "best_round": 0,
            "circuit_name": circuit_name,
            "tag": progress.get("tag"),
            "round_data": None,
            "note": "no successful round; (best) is a copy of the original input",
        }
        with open(output_dir / f"{circuit_name}(best).txt", "w") as f:
            f.write("# (best) = original input  [no round succeeded]\n")
            f.write(json.dumps(body, indent=2, default=str))
            f.write("\n")


def run_circuit(circuit_name: str, input_qasm: Path, rounds_config: list,
                output_dir: Path, tag: str, log_path: Path) -> int:
    """Run all rounds for one circuit. Returns exit code (0 = OK, 1 = all-rounds-failed)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "circuit_name": circuit_name,
        "tag": tag,
        "input_qasm": str(input_qasm),
        "rounds_config": rounds_config,
        "rounds": {},
        "last_successful_round": 0,
    }
    _atomic_write_json(output_dir / "_progress.json", progress)

    current_input = input_qasm
    last_successful_round = 0

    for i, round_params in enumerate(rounds_config, start=1):
        round_qasm = output_dir / f"{circuit_name}({i}).qasm"
        round_txt = output_dir / f"{circuit_name}({i}).txt"
        try:
            result = phasepoly_synthesize(
                circuit_input_path=str(current_input),
                circuit_output_path=str(round_qasm),
                circuit_name=circuit_name,
                label=str(i),
                **round_params,
            )
            _dump_round_txt(round_txt, i, result)
            round_data = dataclasses.asdict(result)
            progress["rounds"][str(i)] = round_data
            last_successful_round = i
            progress["last_successful_round"] = last_successful_round
            _atomic_write_json(output_dir / "_progress.json", progress)
            _append_log(
                log_path,
                f"{circuit_name} round {i} OK | "
                f"cx_red={result.cx_reduction} rz_red={result.rz_reduction} "
                f"t_red={result.t_reduction} total_red={result.total_reduction} "
                f"time={result.total_time:.3f}s "
                f"params={round_params}",
            )
            current_input = round_qasm  # chain: next round reads this round's output
        except PhasepolySynthesisError as e:
            tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
            result = e.result
            round_data = dataclasses.asdict(result)
            round_data["trace"] = tb
            round_data["params"] = round_params
            _dump_round_txt(round_txt, i, result)
            progress["rounds"][str(i)] = round_data
            _atomic_write_json(output_dir / "_progress.json", progress)
            _append_log(
                log_path,
                f"{circuit_name} round {i} ERROR: {e} | "
                f"synth_time={result.synthesis_time:.3f}s total_time={result.total_time:.3f}s",
            )
            # Copy the chained input as (N).qasm so every round has an output
            # file (downstream tooling can rely on the naming convention) and
            # the chain stays unbroken.
            try:
                shutil.copyfile(current_input, round_qasm)
            except Exception as copy_err:
                _append_log(log_path, f"{circuit_name} round {i} copy-fallback failed: {copy_err}")
            # current_input is unchanged: next round still reads the last
            # successful output (or the original input).
        except Exception as e:
            tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
            progress["rounds"][str(i)] = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "trace": tb,
                "params": round_params,
            }
            _atomic_write_json(output_dir / "_progress.json", progress)
            _append_log(log_path, f"{circuit_name} round {i} ERROR: {e}")
            try:
                shutil.copyfile(current_input, round_qasm)
            except Exception as copy_err:
                _append_log(log_path, f"{circuit_name} round {i} copy-fallback failed: {copy_err}")

    _maybe_write_best(output_dir, circuit_name, progress, input_qasm)
    _atomic_write_json(output_dir / "result.json", progress)

    return 0 if last_successful_round > 0 else 1


def _build_arg_parser():
    p = argparse.ArgumentParser(prog="_run_one_circuit", description=__doc__)
    p.add_argument("--circuit-name", required=True)
    p.add_argument("--input-qasm", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--log-path", required=True, type=Path)
    p.add_argument("--rounds-json", required=True,
                   help="JSON-encoded list of round-parameter dicts")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    rounds_config = json.loads(args.rounds_json)
    if not isinstance(rounds_config, list) or not rounds_config:
        print("rounds-json must be a non-empty JSON list", file=sys.stderr)
        return 2
    return run_circuit(
        circuit_name=args.circuit_name,
        input_qasm=args.input_qasm,
        rounds_config=rounds_config,
        output_dir=args.output_dir,
        tag=args.tag,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    sys.exit(main())

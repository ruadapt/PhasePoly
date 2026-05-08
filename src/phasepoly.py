# src.phasepoly
# A new phasepoly synthesis program, created at 2025-08-13

import os
import warnings
from contextlib import contextmanager
from timeit import default_timer as timer
from dataclasses import dataclass, field
from typing import Any, Literal
import src.circuits as circuits
import src.parser as parser
from src.synthesis import (
    get_phasePoly_groups,
    set_gaussian_elim_algorithm,
    synthesize_grouped_phasePoly_in_place,
    synthesize_in_place,
)
from src.qasm_reader import (
    load_circuit_from_qasm,
    get_circuit_info,
)

@dataclass
class PhasepolySynthesisResult:
    circuit_name: str
    label: str
    success: bool = True
    failure_stage: str = ""
    error_type: str = ""
    error: str = ""
    rotation_merging_time: float = 0.0
    circuit_partitioning_time: float = 0.0
    synthesis_time: float = 0.0
    data_io_time: float = 0.0
    total_time: float = 0.0
    input_circuit_info: dict[str, Any] = field(default_factory=dict)
    circuit_info: dict[str, Any] = field(default_factory=dict)
    total_reduction: float = 0.0
    cx_reduction: float = 0.0
    rz_reduction: float = 0.0
    t_reduction: float = 0.0
    warnings: list[str] = field(default_factory=list)


class PhasepolySynthesisError(RuntimeError):
    """Raised when synthesis fails after a partial result has been collected."""

    def __init__(self, message: str, result: PhasepolySynthesisResult):
        super().__init__(message)
        self.result = result


@contextmanager
def _timed(result: PhasepolySynthesisResult, attr: str, accumulate: bool = False):
    """Accumulate elapsed time into result.<attr>. accumulate=True adds, False overwrites."""
    start = timer()
    try:
        yield
    finally:
        elapsed = timer() - start
        if accumulate:
            setattr(result, attr, getattr(result, attr) + elapsed)
        else:
            setattr(result, attr, elapsed)


def _safe_circuit_info(file_path: str, circuit_name: str) -> dict[str, Any]:
    try:
        return get_circuit_info(load_circuit_from_qasm(file_path), circuit_name)
    except Exception as e:
        return {
            "metrics_error": str(e),
            "metrics_error_type": type(e).__name__,
        }


def _run_rotation_merging(circ: circuits.Circuit, mode: Literal["pure_rotation_merging", "advanced_rotation_merging"]):
    if mode == "pure_rotation_merging":
        circ.circuit_wide_rz_floating(pure_rotation_merging=True)
    elif mode == "advanced_rotation_merging":
        circ.circuit_wide_rz_floating(pure_rotation_merging=False)
        circ.transform_cx_h_gates()
        circ.circuit_wide_rz_floating(pure_rotation_merging=False)
        circ.transform_cx_h_gates()
    else:
        raise ValueError(
            f"Invalid rotation_merging_mode '{mode}'. "
            "Must be one of {'pure_rotation_merging', 'advanced_rotation_merging'}."
        )


def phasepoly_synthesize(
    circuit_input_path: str,            # Path to the input circuit
    circuit_output_path: str,           # Path to the output circuit
    method: Literal[
        "row_heap",
        "row_heap_classical_GE",
        "single_block_greedy",
        "single_block_greedy_classical_GE",
        "rotation_merging",
        "pure_rotation_merging",
    ] = "row_heap",
    rotation_merging_mode: Literal["pure_rotation_merging", "advanced_rotation_merging"] = "advanced_rotation_merging",
    heap_size: int = 10000,             # Size of the heap
    ends_checked: int = 10000,          # Number of ends check list
    group_size: int = 1,                # Size of the group
    circuit_name: str = "circuit",      # Name of the circuit
    label: str = "0",                   # Label of the optimization
    gaussian_elim_algorithm: Literal["modified", "classic"] = "modified",  # GE backend
) -> PhasepolySynthesisResult:
    """
    Returns a structured PhasepolySynthesisResult instead of a dynamic dict.
    Raises ValueError on invalid arguments.
    May raise underlying exceptions (e.g., from parser or synthesis) unless caught below.
    """
    valid_methods = {
        "row_heap",
        "row_heap_classical_GE",
        "single_block_greedy",
        "single_block_greedy_classical_GE",
        "pure_rotation_merging",
        "rotation_merging",
    }
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {sorted(valid_methods)}.")

    valid_merge_modes = {"pure_rotation_merging", "advanced_rotation_merging"}
    if rotation_merging_mode not in valid_merge_modes:
        raise ValueError(
            f"Invalid rotation_merging_mode '{rotation_merging_mode}'. "
            f"Must be one of {sorted(valid_merge_modes)}."
        )

    grouped_methods = {"row_heap", "row_heap_classical_GE"}
    if group_size > 1 and method not in grouped_methods:
        raise ValueError(
            "group_size > 1 is only supported with method='row_heap' or "
            f"method='row_heap_classical_GE', got method='{method}'."
        )

    valid_ge_algos = {"modified", "classic"}
    if gaussian_elim_algorithm not in valid_ge_algos:
        raise ValueError(
            f"Invalid gaussian_elim_algorithm '{gaussian_elim_algorithm}'. "
            f"Must be one of {sorted(valid_ge_algos)}."
        )

    result = PhasepolySynthesisResult(circuit_name=circuit_name, label=label)
    total_time_start = timer()
    current_stage = "initialize"

    if method == "row_heap_classical_GE":
        effective_gaussian_elim_algorithm = "classic"
    elif method == "single_block_greedy":
        effective_gaussian_elim_algorithm = "modified"
    elif method == "single_block_greedy_classical_GE":
        effective_gaussian_elim_algorithm = "classic"
    else:
        effective_gaussian_elim_algorithm = gaussian_elim_algorithm

    if method == "pure_rotation_merging":
        effective_rotation_merging_mode = "pure_rotation_merging"
    elif method == "rotation_merging":
        effective_rotation_merging_mode = "advanced_rotation_merging"
    else:
        effective_rotation_merging_mode = rotation_merging_mode

    prev_ge_algo = set_gaussian_elim_algorithm(effective_gaussian_elim_algorithm)
    try:
        # ---- I/O: read ----
        current_stage = "read_input"
        with _timed(result, "data_io_time", accumulate=True):
            result.input_circuit_info = _safe_circuit_info(circuit_input_path, circuit_name)
            circ: circuits.Circuit = parser.read_qasm(circuit_input_path)
            circ.s_and_t_to_rz()

        new_circ = circ.copy()

        # ---- rotation merging pre ----
        current_stage = "rotation_merging_pre"
        with _timed(result, "rotation_merging_time", accumulate=True):
            _run_rotation_merging(new_circ, effective_rotation_merging_mode)

        # ---- phasepoly partition + synth (when applicable) ----
        if method in {
            "row_heap",
            "row_heap_classical_GE",
            "single_block_greedy",
            "single_block_greedy_classical_GE",
        }:
            current_stage = "partition"
            with _timed(result, "circuit_partitioning_time"):
                new_circ.partition_to_phasePoly(size_metric="rotations")

            current_stage = "synthesis"
            with _timed(result, "synthesis_time"):
                if group_size == 1:
                    # Single group synthesis
                    for n in new_circ.get_sequence():
                        if getattr(n, "nodeType", None) == "phasePoly":
                            if method == "single_block_greedy":
                                synthesize_in_place(n, "single_block_greedy")
                            elif method == "single_block_greedy_classical_GE":
                                synthesize_in_place(n, "single_block_greedy_classical_GE")
                            elif method in {"row_heap", "row_heap_classical_GE"}:
                                synthesize_in_place(n, "row_heap", heap_size, ends_checked)
                else:
                    # Multi group synthesis (row_heap-family method guaranteed by validation above)
                    groups = get_phasePoly_groups(new_circ, maxsize=group_size)
                    for g in groups:
                        synthesize_grouped_phasePoly_in_place(g, heap_size, ends_checked, greater_circuit=new_circ)

        # ---- rotation merging post ----
        current_stage = "rotation_merging_post"
        with _timed(result, "rotation_merging_time", accumulate=True):
            _run_rotation_merging(new_circ, effective_rotation_merging_mode)

        # ---- I/O: write ----
        current_stage = "write_output"
        out_dir = os.path.dirname(circuit_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with _timed(result, "data_io_time", accumulate=True):
            parser.write_qasm(new_circ, circuit_output_path)

        # ---- total time (accurate at the end) ----
        result.total_time = timer() - total_time_start

        # ---- Metrics ----
        current_stage = "metrics"
        before_info = result.input_circuit_info
        after_info = _safe_circuit_info(circuit_output_path, circuit_name)
        result.circuit_info = after_info

        total_reduction = 0.0
        cx_reduction = 0.0
        rz_reduction = 0.0
        t_reduction = 0.0
        try:
            total_reduction = (before_info["gates(always weighted)"] - after_info["gates(always weighted)"])
            cx_reduction    = (before_info["weighted_cx"] - after_info["weighted_cx"])
            rz_reduction    = (before_info["rz_gate"] - after_info["rz_gate"])
            t_reduction     = (before_info["t_gate"] - after_info["t_gate"])
        except KeyError as e:
            print(f"KeyError: Missing expected key in circuit info: {e}")

        result.total_reduction = total_reduction
        result.cx_reduction = cx_reduction
        result.rz_reduction = rz_reduction
        result.t_reduction = t_reduction

        if any(x < 0 for x in (result.t_reduction, result.rz_reduction, result.cx_reduction, result.total_reduction)):
            msg = (
                f"{circuit_name} has negative reduction "
                f"(total/cx/rz/t = {result.total_reduction}/{result.cx_reduction}/{result.rz_reduction}/{result.t_reduction})"
            )
            warnings.warn(msg)
            result.warnings.append(msg)

        return result
    except Exception as e:
        result.success = False
        result.failure_stage = current_stage
        result.error_type = type(e).__name__
        result.error = str(e)
        result.total_time = timer() - total_time_start
        if not result.input_circuit_info:
            result.input_circuit_info = _safe_circuit_info(circuit_input_path, circuit_name)
        if os.path.exists(circuit_output_path):
            result.circuit_info = _safe_circuit_info(circuit_output_path, circuit_name)
        result.warnings.append(f"failed: {result.error_type}: {result.error}")
        raise PhasepolySynthesisError(
            f"{circuit_name} phasepoly synthesis failed: {result.error}", result
        ) from e
    finally:
        set_gaussian_elim_algorithm(prev_ge_algo)


def rotation_merging(
    circuit_input_path: str,
    circuit_output_path: str,
    method: Literal["pure_rotation_merging", "advanced_rotation_merging"] = "pure_rotation_merging",
) -> None:
    """Standalone rotation-merging entry. Delegates to _run_rotation_merging for the merge logic."""
    circ: circuits.Circuit = parser.read_qasm(circuit_input_path)
    circ.s_and_t_to_rz()
    _run_rotation_merging(circ, method)
    parser.write_qasm(circ, circuit_output_path)


def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m src.phasepoly",
        description="Phase-polynomial circuit synthesis on a QASM file.",
    )
    p.add_argument("input", help="Input QASM path")
    p.add_argument("output", help="Output QASM path")
    p.add_argument(
        "-m", "--method",
        choices=[
            "row_heap",
            "row_heap_classical_GE",
            "single_block_greedy",
            "single_block_greedy_classical_GE",
            "pure_rotation_merging",
            "rotation_merging",
        ],
        default="row_heap",
        help="Synthesis method (default: row_heap)",
    )
    p.add_argument(
        "--rotation-merging-mode",
        choices=["pure_rotation_merging", "advanced_rotation_merging"],
        default="advanced_rotation_merging",
        help="Rotation-merging mode (default: advanced_rotation_merging)",
    )
    p.add_argument("--heap-size", type=int, default=10000, help="Row-heap A* buffer size (default: 10000)")
    p.add_argument("--ends-checked", type=int, default=10000, help="Max number of terminal states to collect (default: 10000)")
    p.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Multi-block group size; >1 requires method=row_heap or row_heap_classical_GE (default: 1)",
    )
    p.add_argument("--circuit-name", default="circuit", help="Label used in result.circuit_name (default: circuit)")
    p.add_argument("--label", default="0", help="Free-form label echoed back in the result (default: 0)")
    p.add_argument(
        "--gaussian-elim-algorithm",
        choices=["modified", "classic"],
        default="modified",
        help="Gaussian elimination backend: 'modified' (current lookahead, default) or "
             "'classic' (the historical greedy algorithm from commit 896c33195a).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress printing the result summary")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    result = phasepoly_synthesize(
        circuit_input_path=args.input,
        circuit_output_path=args.output,
        method=args.method,
        rotation_merging_mode=args.rotation_merging_mode,
        heap_size=args.heap_size,
        ends_checked=args.ends_checked,
        group_size=args.group_size,
        circuit_name=args.circuit_name,
        label=args.label,
        gaussian_elim_algorithm=args.gaussian_elim_algorithm,
    )
    if not args.quiet:
        print(result)
    return result


if __name__ == "__main__":
    main()

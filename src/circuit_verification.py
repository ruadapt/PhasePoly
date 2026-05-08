# src/circuit_verification.py
#
# ============================================================================
# CIRCUIT EQUIVALENCE VERIFICATION -- HOW TO USE
# ============================================================================
#
# This module is the single source of truth for verifying that two QASM
# circuits compute the same unitary. The thin command-line wrapper lives at
# `scripts/verify_circuits.py`; for batch / shell usage prefer the CLI.
#
# ----------------------------------------------------------------------------
# WHAT THIS MODULE EXPORTS
# ----------------------------------------------------------------------------
#
# Verification methods (each returns `(status, detail)`; statuses are
# "ok" | "skipped" | "timeout" | "error"):
#
#     qiskit_equivalence_verification(original_path, compared_path,
#                                     timeout=DEFAULT_TIMEOUT)
#         Qiskit Operator.equiv. Gated to circuits with <= QISKIT_MAX_QUBITS
#         (= 8) qubits; larger circuits return ("skipped", ...).
#
#     mqt_qcec_verification(original_path, compared_path,
#                           timeout=DEFAULT_TIMEOUT)
#         mqt.qcec.verify formal verification. No qubit cap.
#
# Drivers:
#
#     verify_pair(original_path, compared_path,
#                 methods=ALL_METHODS, timeout=DEFAULT_TIMEOUT) -> dict
#         Run the requested methods on a single pair; returns a record
#         {original, compared, <method>: {status, detail, elapsed}, ...}.
#
#     verify_folder_pair(keys, original_folder, compared_folder,
#                        methods=ALL_METHODS, timeout=DEFAULT_TIMEOUT,
#                        log_path=None) -> list[dict]
#         For each key in `keys`, locate one .qasm file in each folder
#         (see PAIRING below), verify the pair, print the result, and
#         optionally Tee stdout/stderr to `log_path`.
#
# Pairing helpers:
#
#     pair_files_by_keys(keys, folder) -> {key: matching_file_path or None}
#         Build the key->file map used by verify_folder_pair.
#
# Misc:
#
#     load_circuit(path) -> QuantumCircuit
#     get_num_qubits(path) -> int
#     ALL_METHODS, METHOD_QISKIT, METHOD_QCEC, DEFAULT_TIMEOUT,
#     QISKIT_MAX_QUBITS, DEFAULT_CIRCUIT_KEYS
#
# ----------------------------------------------------------------------------
# QUICK PYTHON USAGE
# ----------------------------------------------------------------------------
#
#     from src.circuit_verification import (
#         qiskit_equivalence_verification, mqt_qcec_verification,
#         verify_pair, verify_folder_pair, DEFAULT_CIRCUIT_KEYS,
#     )
#
#     # Single pair, both methods, 5-minute budget per method:
#     #   original = reference circuit, compared = optimized PPS output
#     rec = verify_pair(
#         "benchmarks/general/barenco_tof_4.qasm",
#         "main_results/phasepoly_results/pps_best/barenco_tof_4(7).qasm",
#         timeout=300,
#     )
#     print(rec["qcec"]["status"], rec["qcec"]["detail"])
#
#     # Folder vs folder, qcec only, 60s budget, log under results/verification/:
#     verify_folder_pair(
#         keys=DEFAULT_CIRCUIT_KEYS,
#         original_folder="benchmarks/general",
#         compared_folder="main_results/phasepoly_results/pps_best",
#         methods=["qcec"], timeout=60,
#         log_path="results/verification/run.log",
#     )
#
# ----------------------------------------------------------------------------
# WHY THE TIMEOUT MATTERS
# ----------------------------------------------------------------------------
#
# `mqt.qcec.verify` is a C++ extension whose work happens in a thread that
# ignores Python signals -- a plain `signal.alarm` cannot kill it. Some
# circuits (e.g. `mod_adder_1024`) will run for hours unless killed.
#
# To cap each verification, every call runs in a fresh child process spawned
# via `multiprocessing` (spawn context). After `timeout` seconds the parent
# does `terminate -> join(5) -> kill` and reports ("timeout", ...). Default
# is 300 s (5 min); pass `timeout=N` to override.
#
# ----------------------------------------------------------------------------
# PAIRING (folder vs folder)
# ----------------------------------------------------------------------------
#
# `pair_files_by_keys(keys, folder)` assigns each .qasm file in `folder` to
# at most one key, using "longest matching key wins":
#
#   - File `barenco_tof_5(7).qasm` matches BOTH `tof_5` and `barenco_tof_5`,
#     but `barenco_tof_5` is the longer key, so the file binds there.
#   - File `tof_5(7).qasm` matches only `tof_5`.
#   - File `optimized_adder_8.qasm`, `adder_8_optimized.qasm`,
#     `adder_8(optimized).qasm` all bind to key `adder_8`.
#
# When several files all bind to the same key, the one with the shortest
# basename wins (closest match). Files matching no key are ignored.
# ============================================================================

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from typing import Iterable

from qiskit import QuantumCircuit


QISKIT_MAX_QUBITS = 8
DEFAULT_TIMEOUT = 300
METHOD_QISKIT = "qiskit"
METHOD_QCEC = "qcec"
ALL_METHODS = (METHOD_QISKIT, METHOD_QCEC)


# ---------------------------------------------------------------------------
# Circuit loading
# ---------------------------------------------------------------------------

def load_circuit(path: str) -> QuantumCircuit:
    with open(path, "r") as f:
        return QuantumCircuit.from_qasm_str(f.read())


def get_num_qubits(path: str) -> int:
    return load_circuit(path).num_qubits


# ---------------------------------------------------------------------------
# Subprocess workers (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _qiskit_worker(orig_path: str, comp_path: str, q: "mp.Queue") -> None:
    try:
        from qiskit.quantum_info import Operator
        orig = load_circuit(orig_path)
        comp = load_circuit(comp_path)
        result = Operator(orig).equiv(Operator(comp))
        q.put(("ok", str(bool(result))))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"))


def _qcec_worker(orig_path: str, comp_path: str, q: "mp.Queue") -> None:
    try:
        from mqt import qcec
        orig = load_circuit(orig_path)
        comp = load_circuit(comp_path)
        result = qcec.verify(orig, comp)
        q.put(("ok", str(result.equivalence)))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"))


def _run_with_timeout(target, args: tuple, timeout: int) -> tuple[str, str]:
    # Prefer "fork" so this works inside Jupyter notebooks and scripts that
    # lack an `if __name__ == "__main__"` guard. "spawn" would re-import the
    # parent module in the child, which fails for notebook / ad-hoc scripts.
    # On non-POSIX systems (no fork available) fall back to "spawn".
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(*args, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join(2)
        return ("timeout", f"timeout after {timeout}s")
    if not q.empty():
        return q.get()
    return ("error", f"worker exited without result (exitcode={p.exitcode})")


# ---------------------------------------------------------------------------
# Verification methods
# ---------------------------------------------------------------------------

def qiskit_equivalence_verification(
    original_path: str,
    compared_path: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, str]:
    """Qiskit Operator equivalence (gated by QISKIT_MAX_QUBITS).

    Returns (status, detail) where status is one of:
        "ok" | "skipped" | "timeout" | "error"
    """
    try:
        n = get_num_qubits(original_path)
    except Exception as e:
        return ("error", f"failed to load original: {type(e).__name__}: {e}")
    if n > QISKIT_MAX_QUBITS:
        return ("skipped", f"qubits {n} > {QISKIT_MAX_QUBITS}")
    return _run_with_timeout(_qiskit_worker, (original_path, compared_path), timeout)


def mqt_qcec_verification(
    original_path: str,
    compared_path: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, str]:
    """mqt.qcec formal verification with subprocess timeout."""
    return _run_with_timeout(_qcec_worker, (original_path, compared_path), timeout)


# ---------------------------------------------------------------------------
# Folder pairing
# ---------------------------------------------------------------------------

def _basename_no_ext(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]


def pair_files_by_keys(keys: Iterable[str], folder: str) -> dict[str, str | None]:
    """Map each key to the best-matching .qasm file in `folder`.

    Disambiguation rule: a file is assigned to the LONGEST key that appears as
    a substring of its basename (sans extension). Each file binds to at most
    one key. If multiple files bind to the same key, the one with the shortest
    basename wins (closest match).
    """
    keys = list(keys)
    if not os.path.isdir(folder):
        return {k: None for k in keys}

    files = sorted(f for f in os.listdir(folder) if f.endswith(".qasm"))
    keys_by_len = sorted(keys, key=lambda k: -len(k))

    assignments: dict[str, list[str]] = {k: [] for k in keys}
    for f in files:
        name = _basename_no_ext(f)
        for k in keys_by_len:
            if k in name:
                assignments[k].append(f)
                break

    out: dict[str, str | None] = {}
    for k in keys:
        candidates = assignments[k]
        if not candidates:
            out[k] = None
        else:
            best = min(candidates, key=lambda f: len(_basename_no_ext(f)))
            out[k] = os.path.join(folder, best)
    return out


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for stream in self._streams:
            stream.write(s)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


# ---------------------------------------------------------------------------
# Folder-by-folder driver
# ---------------------------------------------------------------------------

def verify_pair(
    original_path: str,
    compared_path: str,
    methods: Iterable[str] = ALL_METHODS,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Run the requested methods on a single circuit pair."""
    record: dict = {
        "original": original_path,
        "compared": compared_path,
    }
    for method in methods:
        t0 = time.perf_counter()
        if method == METHOD_QISKIT:
            status, detail = qiskit_equivalence_verification(original_path, compared_path, timeout)
        elif method == METHOD_QCEC:
            status, detail = mqt_qcec_verification(original_path, compared_path, timeout)
        else:
            status, detail = ("error", f"unknown method: {method}")
        record[method] = {
            "status": status,
            "detail": detail,
            "elapsed": time.perf_counter() - t0,
        }
    return record


def verify_folder_pair(
    keys: Iterable[str],
    original_folder: str,
    compared_folder: str,
    methods: Iterable[str] = ALL_METHODS,
    timeout: int = DEFAULT_TIMEOUT,
    log_path: str | None = None,
) -> list[dict]:
    """Verify every key pair across two folders, printing & optionally logging."""
    keys = list(keys)
    methods = list(methods)

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
    else:
        log_file = None
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    if log_file is not None:
        sys.stdout = _Tee(saved_stdout, log_file)
        sys.stderr = _Tee(saved_stderr, log_file)

    results: list[dict] = []
    try:
        original_map = pair_files_by_keys(keys, original_folder)
        compared_map = pair_files_by_keys(keys, compared_folder)

        start = datetime.now()
        print(f"[verify_folder_pair] start: {start:%Y-%m-%d %H:%M:%S}")
        print(f"  original_folder: {original_folder}")
        print(f"  compared_folder: {compared_folder}")
        print(f"  methods: {methods}    timeout: {timeout}s    keys: {len(keys)}")

        for key in keys:
            orig = original_map.get(key)
            comp = compared_map.get(key)
            print(f"\n=== {key} ===")
            if orig is None:
                print(f"  [skip] no original file matches key in {original_folder}")
                results.append({"key": key, "skipped": "no_original"})
                continue
            if comp is None:
                print(f"  [skip] no compared file matches key in {compared_folder}")
                results.append({"key": key, "skipped": "no_compared"})
                continue
            print(f"  original: {orig}")
            print(f"  compared: {comp}")
            try:
                n = get_num_qubits(orig)
                print(f"  qubits:   {n}")
            except Exception as e:
                print(f"  qubits:   <load failed: {type(e).__name__}: {e}>")

            record = verify_pair(orig, comp, methods=methods, timeout=timeout)
            record["key"] = key
            for method in methods:
                m = record[method]
                print(f"  {method:>6}: {m['status']:<8} {m['detail']}  ({m['elapsed']:.2f}s)")
            results.append(record)

        end = datetime.now()
        print(f"\n[verify_folder_pair] end: {end:%Y-%m-%d %H:%M:%S}    elapsed: {end - start}")
    finally:
        if log_file is not None:
            sys.stdout, sys.stderr = saved_stdout, saved_stderr
            log_file.close()
            print(f"[verify_folder_pair] log saved to: {log_path}")

    return results


DEFAULT_CIRCUIT_KEYS: list[str] = [
    "adder_8",
    "barenco_tof_3",
    "barenco_tof_4",
    "barenco_tof_5",
    "barenco_tof_10",
    "csla_mux_3",
    "gf2^4_mult",
    "gf2^5_mult",
    "grover_5",
    "ham15-high",
    "ham15-low",
    "ham15-med",
    "hwb6",
    "mod_adder_1024",
    "mod_mult_55",
    "mod_red_21",
    "mod5_4",
    "qaoa_n8_p4",
    "qaoa_n10_p4",
    "qcla_adder_10",
    "qcla_com_7",
    "qcla_mod_7",
    "rc_adder_6",
    "tof_3",
    "tof_4",
    "tof_5",
    "tof_10",
    "vbe_adder_3",
    "shor_15_7",
]

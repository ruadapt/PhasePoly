# Verification

`src/circuit_verification.py` checks that two QASM circuits compute the same unitary. Two backends:

| Backend | Function | When it fires | Cost |
|---|---|---|---|
| Qiskit `Operator.equiv` | `qiskit_equivalence_verification` | circuits with **≤ 8 qubits** (else returns `("skipped", ...)`) | exponential in qubits; only practical for the paper figures |
| MQT QCEC | `mqt_qcec_verification` | any size; the workhorse for benchmarks | seconds for medium circuits, can hit the timeout on >25-qubit, deep ones |

Both return `(status, detail)` where `status ∈ {"ok", "skipped", "timeout", "error"}`.

## Verifying one pair

```python
from src.circuit_verification import verify_pair

rec = verify_pair(
    "benchmarks/general/adder_8.qasm",
    "/tmp/adder_8_opt.qasm",
    methods=["qiskit", "qcec"],   # or just ["qcec"] for >8-qubit circuits
    timeout=120,                  # per-method budget in seconds
)

# rec["qcec"]   = {"status": "ok", "detail": EquivalenceCriterion.equivalent, "elapsed": 0.07}
# rec["qiskit"] = {"status": "skipped", "detail": "...", "elapsed": 0.0}   # adder_8 has 24 qubits
```

## Verifying a whole folder

```python
from src.circuit_verification import verify_folder_pair, DEFAULT_CIRCUIT_KEYS

results = verify_folder_pair(
    keys=DEFAULT_CIRCUIT_KEYS,
    original_folder="benchmarks/general",
    compared_folder="results/my_run/phasepoly_best_my_run",
    methods=["qcec"],
    timeout=900,
    log_path="results/verification/my_run.log",
)
```

`DEFAULT_CIRCUIT_KEYS` is the 27-circuit list used in the paper. `keys` does substring matching against filenames in each folder, so `"adder_8"` matches both `adder_8.qasm` and `adder_8(best).qasm`.

## Notebook walkthrough

[`../circuit_verification_demo.ipynb`](../circuit_verification_demo.ipynb) verifies the four paper figures, one named benchmark, and the full folder. Use it as a template.

## Runtime guidance

| Circuit shape | Recommended methods | Typical QCEC time |
|---|---|---|
| Paper figures (≤ 5 qubits) | `["qiskit", "qcec"]` | < 0.1 s |
| Small benchmarks (≤ 12 qubits) | `["qiskit", "qcec"]` | < 1 s |
| Medium benchmarks (13–22 qubits) | `["qcec"]` | 1–30 s |
| Large benchmarks (>22 qubits, deep) | `["qcec"]` with `timeout=900` | 30 s – minutes; sometimes hits timeout |

Each `verify_pair` call runs the methods in subprocesses with the timeout — a rogue 100-minute QCEC won't hang your script.

## CLI wrapper

For shell usage there's `scripts/verify_circuits.py`; it takes the same `--keys`, `--original-folder`, `--compared-folder`, `--methods`, `--timeout`, `--log-path` and prints to stdout.

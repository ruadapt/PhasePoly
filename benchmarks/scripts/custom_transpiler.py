# benchmarks.scripts.custom_transpiler
# CCX (Toffoli) decomposition into {CX, H, RZ} basis.
# Used by recursive_circuits_gen.py to produce QASM benchmarks.

from math import pi
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps
import os


class CustomTranspiler:
    """Decomposes CCX gates into the {CX, H, RZ} gate set."""

    def add_decompose_ccx(self, qc: QuantumCircuit, control_1: int, control_2: int,
                          target: int, is_flip=False, reverse=False) -> QuantumCircuit:
        """
        Append a CCX-equivalent sequence of {H, CX, RZ} gates to qc.

        Args:
            is_flip:   negate all RZ angles (used by certain decomposition strategies).
            reverse:   emit gates in reverse order (used by hat-structure uncompute pass).
        """
        flip = -1 if is_flip else 1

        if not reverse:
            qc.h(target)
            qc.cx(control_2, target)
            qc.rz(flip * (-pi / 4), target)
            qc.cx(control_1, target)
            qc.rz(flip * (pi / 4), target)
            qc.cx(control_2, target)
            qc.rz(flip * (-pi / 4), target)
            qc.rz(flip * (pi / 4), control_2)
            qc.cx(control_1, target)
            qc.rz(flip * (pi / 4), target)
            qc.cx(control_1, control_2)
            qc.h(target)
            qc.rz(flip * (-pi / 4), control_2)
            qc.rz(flip * (pi / 4), control_1)
            qc.cx(control_1, control_2)
        else:
            qc.cx(control_1, control_2)
            qc.rz(flip * (pi / 4), control_1)
            qc.rz(flip * (-pi / 4), control_2)
            qc.h(target)
            qc.cx(control_1, control_2)
            qc.rz(flip * (pi / 4), target)
            qc.cx(control_1, target)
            qc.rz(flip * (pi / 4), control_2)
            qc.rz(flip * (-pi / 4), target)
            qc.cx(control_2, target)
            qc.rz(flip * (pi / 4), target)
            qc.cx(control_1, target)
            qc.rz(flip * (-pi / 4), target)
            qc.cx(control_2, target)
            qc.h(target)

        return qc


def qasm_write_to_file(qc: QuantumCircuit, folder_name: str, file_name: str):
    """Write qc to <folder_name>/<file_name>.qasm, creating the folder if needed."""
    file_path = f"{folder_name}/{file_name}.qasm"
    os.makedirs(folder_name, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(dumps(qc).encode())
    print(f"QuantumCircuit written to {file_path}")


if __name__ == "__main__":
    ct = CustomTranspiler()
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    print(qc.draw())
    for instr, qargs, cargs in qc.data:
        if instr.name == "ccx":
            ctrl1, ctrl2, tgt = qargs
            qc.data.pop(0)
            ct.add_decompose_ccx(qc, ctrl1._index, ctrl2._index, tgt._index, is_flip=False)
    print(qc.draw())

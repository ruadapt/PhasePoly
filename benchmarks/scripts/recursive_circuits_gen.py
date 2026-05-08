# benchmarks.scripts.recursive_circuits_gen
# Generates multi-controlled Toffoli benchmark circuits using three decomposition
# strategies: v_chain, hat_structure, and recursive.
#
# Run from the project root:
#   python -m benchmarks.scripts.recursive_circuits_gen
#
# Output QASM files land in benchmarks/mcx_n/ by convention.

from typing import List, Optional, Tuple
from qiskit import QuantumCircuit, transpile
from mqt import qcec
from benchmarks.scripts.custom_transpiler import CustomTranspiler


class RecursiveCircuitsGen:
    """
    Generates multi-controlled Toffoli gates (n >= 3) using three strategies:
      1. v_chain_mtoffoli       — linear-depth V-chain of CCX gates
      2. hat_structure_mtoffoli — hat-structure (two upward/downward passes)
      3. recursive_mtoffoli     — recursive decomposition for n >= 5
    """

    def __init__(self):
        self.qubits_num = 0

    def _prepare_circuit(
        self,
        qc: Optional[QuantumCircuit],
        control_num: int,
        controls: List[int],
        target: int,
        ancillas: List[int]
    ) -> QuantumCircuit:
        """Validate inputs and return a QuantumCircuit with sufficient qubits."""
        if control_num < 3:
            raise ValueError("control_num must be at least 3.")
        if len(controls) != control_num:
            raise ValueError("Length of controls must be control_num.")
        if len(ancillas) != control_num - 2:
            raise ValueError("Length of ancillas must be control_num - 2.")
        if target in controls or target in ancillas:
            raise ValueError("Target qubit cannot be one of the controls or ancillas.")
        if any(ctrl in ancillas for ctrl in controls):
            raise ValueError("Control qubits cannot overlap with ancilla qubits.")

        max_index = max(controls + [target] + ancillas)
        if qc is None:
            qc = QuantumCircuit(max_index + 1)
        elif qc.num_qubits < max_index + 1:
            raise ValueError("Provided QuantumCircuit does not have enough qubits.")
        return qc

    def _apply_decomposition(self, qc: QuantumCircuit, decompose_method: Optional[str], control_num: int) -> QuantumCircuit:
        """Optionally decompose the circuit into a specified gate basis."""
        custom_transpiler = CustomTranspiler()
        if decompose_method == "qiskit":
            qc = transpile(qc, basis_gates=['cx', 'h', 'rz', 'x'], optimization_level=0)
        elif decompose_method == "custom_v_chain":
            final_circuit = QuantumCircuit(qc.num_qubits)
            for idx, ctrl1, ctrl2, tgt in self._get_all_toffoli_gate_sequence(qc):
                is_flip = control_num <= idx <= 2 * control_num - 3
                custom_transpiler.add_decompose_ccx(final_circuit, ctrl1, ctrl2, tgt, is_flip=is_flip)
            qc = final_circuit
        elif decompose_method == "custom_hat_structure":
            final_circuit = QuantumCircuit(qc.num_qubits)
            for idx, ctrl1, ctrl2, tgt in self._get_all_toffoli_gate_sequence(qc):
                is_flip = (control_num <= idx <= 2 * control_num - 3 or
                           3 * control_num - 5 <= idx <= 4 * control_num - 8)
                custom_transpiler.add_decompose_ccx(final_circuit, ctrl1, ctrl2, tgt, is_flip=is_flip)
            qc = final_circuit
        elif decompose_method == "custom_hat_structure_reverse":
            final_circuit = QuantumCircuit(qc.num_qubits)
            for idx, ctrl1, ctrl2, tgt in self._get_all_toffoli_gate_sequence(qc):
                is_flip = (control_num <= idx <= 2 * control_num - 3 or
                           3 * control_num - 5 <= idx <= 4 * control_num - 8)
                custom_transpiler.add_decompose_ccx(final_circuit, ctrl1, ctrl2, tgt,
                                                    is_flip=is_flip, reverse=is_flip)
            qc = final_circuit
        return qc

    def _get_all_toffoli_gate_sequence(self, qc: QuantumCircuit) -> List[Tuple[int, int, int, int]]:
        """Return (1-indexed position, ctrl1, ctrl2, target) for every CCX in qc."""
        result = []
        for index, (instr, qargs, _) in enumerate(qc.data, start=1):
            if instr.name == "ccx":
                ctrl1, ctrl2, tgt = qargs
                result.append((index, ctrl1._index, ctrl2._index, tgt._index))
        return result

    def v_chain_mtoffoli(
        self,
        qc: Optional[QuantumCircuit],
        control_num: int,
        controls: List[int],
        target: int,
        ancillas: List[int],
        decompose_method: Optional[str] = None
    ) -> QuantumCircuit:
        """
        n-controlled Toffoli via V-chain of CCX gates.

        Compute phase by chaining controls into ancillas, flip the target with the
        last control + final ancilla, then uncompute in reverse.
        """
        qc = self._prepare_circuit(qc, control_num, controls, target, ancillas)

        # Compute phase
        qc.ccx(controls[0], controls[1], ancillas[0])
        for i in range(2, control_num - 1):
            qc.ccx(controls[i], ancillas[i - 2], ancillas[i - 1])

        # Final Toffoli
        qc.ccx(controls[-1], ancillas[-1], target)

        # Uncompute phase (reverse the compute chain)
        for i in range(control_num - 2, 1, -1):
            qc.ccx(controls[i], ancillas[i - 2], ancillas[i - 1])
        qc.ccx(controls[0], controls[1], ancillas[0])

        return self._apply_decomposition(qc, decompose_method, control_num)

    def hat_structure_mtoffoli(
        self,
        qc: Optional[QuantumCircuit],
        control_num: int,
        controls: List[int],
        target: int,
        ancillas: List[int],
        decompose_method: Optional[str] = None
    ) -> QuantumCircuit:
        """
        n-controlled Toffoli via hat-structure: two upward/downward chain passes
        around a central V-chain, enabling T-count reduction after phase-poly synthesis.
        """
        qc = self._prepare_circuit(qc, control_num, controls, target, ancillas)

        # Left upward chain
        qc.ccx(controls[-1], ancillas[-1], target)
        for i in range(control_num - 2, 1, -1):
            qc.ccx(controls[i], ancillas[i - 2], ancillas[i - 1])

        # Central V-chain
        qc.ccx(controls[0], controls[1], ancillas[0])
        for i in range(2, control_num - 1):
            qc.ccx(controls[i], ancillas[i - 2], ancillas[i - 1])
        qc.ccx(controls[-1], ancillas[-1], target)

        # Uncompute
        for i in range(control_num - 2, 1, -1):
            qc.ccx(controls[i], ancillas[i - 2], ancillas[i - 1])
        qc.ccx(controls[0], controls[1], ancillas[0])

        # Right downward chain
        for i in range(2, control_num - 1):
            qc.ccx(controls[i], ancillas[i - 2], ancillas[i - 1])

        return self._apply_decomposition(qc, decompose_method, control_num)

    def recursive_mtoffoli(
        self,
        qc: Optional[QuantumCircuit],
        control_num: int,
        controls: List[int],
        target: int,
        ancillas: List[int],
        all_ancillas: Optional[int],
        decompose_method: Optional[str] = None
    ) -> QuantumCircuit:
        """
        n-controlled Toffoli via recursive decomposition (n >= 5).
        Reduces to C(n-1)X + CCX + C(n-1)X + CCX; base case at n=4 uses hat_structure.
        """
        if control_num < 3:
            raise ValueError("control_num must be at least 5.")
        if len(controls) != control_num:
            raise ValueError("Length of controls must be control_num.")
        if len(ancillas) != 1:
            raise ValueError("Length of ancillas must be 1.")
        if target in controls or target in ancillas:
            raise ValueError("Target qubit cannot be one of the controls or ancillas.")
        if any(ctrl in ancillas for ctrl in controls):
            raise ValueError("Control qubits cannot overlap with ancilla qubits.")

        max_index = max(controls + [target] + ancillas)
        if qc is None:
            qc = QuantumCircuit(max_index + 1)
        elif qc.num_qubits < max_index + 1:
            raise ValueError("Provided QuantumCircuit does not have enough qubits.")

        self.qubits_num = qc.num_qubits

        if all_ancillas is None:
            all_ancillas = controls + ancillas + [target]
        accepted_ancillas = sorted(list(set(all_ancillas) - set(controls + [target])))

        if control_num == 4:
            hat_ancillas = accepted_ancillas[:control_num - 2]
            return self.hat_structure_mtoffoli(qc, control_num, controls, target, hat_ancillas, None)

        subcirc1 = self.recursive_mtoffoli(
            QuantumCircuit(self.qubits_num), control_num - 1,
            controls[:-1], ancillas[0], [controls[-1]], all_ancillas, None
        )
        qc.compose(subcirc1, qubits=range(qc.num_qubits), inplace=True)
        qc.ccx(controls[-1], ancillas[0], target)

        subcirc2 = self.recursive_mtoffoli(
            QuantumCircuit(self.qubits_num), control_num - 1,
            controls[:-1], ancillas[0], [controls[-1]], all_ancillas, None
        )
        qc.compose(subcirc2, qubits=range(qc.num_qubits), inplace=True)
        qc.ccx(controls[-1], ancillas[0], target)

        return qc


if __name__ == "__main__":
    rcg = RecursiveCircuitsGen()
    n = 10
    controls = list(range(n))
    ancillas = list(range(n, 2 * n - 2))
    target = 2 * n - 2

    qc = rcg.hat_structure_mtoffoli(None, n, controls, target, ancillas)
    print(qc.draw())

    ref_file = "./benchmarks/general/barenco_tof_10.qasm"
    qc_ref = QuantumCircuit.from_qasm_file(ref_file)
    result = qcec.verify(qc_ref, qc)
    print(f"Equivalence check: {result.equivalence}")

# verification_checker.py

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal
from mqt import qcec
import numpy as np

# unitary matrix verification
# ATOL_DEFAULT = 1e-8 | RTOL_DEFAULT = 1e-5
def unitary_matrix_verification(original_circuit:QuantumCircuit, compared_circuit:QuantumCircuit)->bool:
    if circuit_size_check(original_circuit):
        original_operator = Operator(original_circuit)
        compared_operator = Operator(compared_circuit)
        return matrix_equal(original_operator.data, compared_operator.data) # ignore_phase=False
    else:
        print("Circuit size is too large for unitary matrix verification")
    
# qiskit equivalence verification
# accept the global phase difference
def qiskit_equivalence_verification(original_circuit:QuantumCircuit, compared_circuit:QuantumCircuit)->bool:
    if circuit_size_check(original_circuit):
        original_operator = Operator(original_circuit)
        compared_operator = Operator(compared_circuit)
        return original_operator.equiv(compared_operator)
    else:
        print("Circuit size is too large for qiskit equivalence verification")

# mqt qcec verification
# multiple methods or formal verification
def mqt_qcec_verification(original_circuit:QuantumCircuit, compared_circuit:QuantumCircuit)->str:
    if circuit_size_check(original_circuit, max_qubits=40):
        result = qcec.verify(original_circuit, compared_circuit)
        return str(result.equivalence) # equivalent | equivalent_up_to_global_phase | equivalent_up_to_phase | no_information | not_equivalent | probably_equivalent | probably
    else:
        return None

def circuit_size_check(circuit:QuantumCircuit, max_qubits:int=12)->bool:
    if circuit.num_qubits > max_qubits:
        print("The number of qubits is " + str(circuit.num_qubits) + " which is larger than the maximum number of qubits " + str(max_qubits))
        return False
    else:
        print("The number of qubits is " + str(circuit.num_qubits))
        return True

def is_linearly_independent(
    matrix: np.ndarray,
    target_row: np.ndarray,
    print_ranks: bool = False
) -> bool:
    """
    Efficiently check if target_row is linearly independent from matrix rows over GF(2),
    by comparing matrix rank before and after appending target_row.

    Parameters:
        matrix (np.ndarray): Binary matrix (GF(2)) with shape (n, m).
        target_row (np.ndarray): Binary row vector with shape (m,).
        print_ranks (bool): If True, print the rank values for debugging.

    Returns:
        bool: True if target_row is linearly independent (should abandon state),
              False if linearly dependent (can be generated from matrix rows).
    """

    # Edge case: empty matrix means trivially dependent
    #  (if the matrix is size 0, then the target row is also size 0. 
    #   As there is only 1 possible size 0 row, the row is dependent)
    if matrix.size == 0:
        return False

    # Stack matrix and target row
    combined = np.vstack([matrix, target_row])

    # Compute ranks over GF(2)
    rank_before = np.linalg.matrix_rank(matrix % 2)
    rank_after = np.linalg.matrix_rank(combined % 2)

    if print_ranks:
        print(f"Rank before: {rank_before}, Rank after adding target: {rank_after}")

    # If rank increases, then target_row is independent
    return rank_after > rank_before


if __name__ == "__main__":

    # Given binary matrix (mod 2 arithmetic)
    matrix = np.array([
        [1, 0, 0, 1],
        [1, 1, 0, 1],
        [0, 0, 1, 1]
    ])
    target_row = np.array([1, 1, 1, 1])
    print(is_linearly_independent(matrix, target_row, print_ranks=True))
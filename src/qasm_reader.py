# src.qasm_reader

from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.circuit.library import XGate, HGate, RZGate, CXGate
import numpy as np
import sympy as sp
from fractions import Fraction
from typing import Tuple
import re, os
from math import isclose, pi

def load_circuit_from_qasm(file_path: str) -> QuantumCircuit:
    # Read QASM from a file and create a quantum circuit
    with open(file_path, 'r') as file:
        qasm_str = file.read()
    return QuantumCircuit.from_qasm_str(qasm_str)

def get_gate_count(qc: QuantumCircuit, gate_name: str) -> int:
    return sum(1 for instr, _, _ in qc.data if instr.name == gate_name)

_T_ANGLE_TOL = 1e-8
# Pre-compute the four canonical T angles for fast comparison.
_T_ANGLES = (pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4)


def _rz_angle(instr) -> float | None:
    """Return the Rz angle in radians if instr is an Rz-like gate, else None."""
    if instr.name != "rz" and not isinstance(instr, RZGate):
        return None
    if not instr.params:
        return None
    try:
        return float(instr.params[0])
    except (TypeError, ValueError):
        return None


def _is_t_angle(angle: float, tol: float = _T_ANGLE_TOL) -> bool:
    """True if angle (radians) equals ±π/4 or ±3π/4 within tol.
    Note: ±3π/4 is counted as ONE T gate (matches historical behavior)."""
    return any(isclose(angle, a, abs_tol=tol) for a in _T_ANGLES)


def get_t_gate_count(qc: QuantumCircuit) -> int:
    t_count = 0
    for instr, _, _ in qc.data:
        if instr.name == "t" or instr.name == "tdg":
            t_count += 1
            continue
        angle = _rz_angle(instr)
        if angle is not None and _is_t_angle(angle):
            t_count += 1
    return t_count

def get_weighted_depth(circuit: QuantumCircuit, swap_weight: int = 3) -> int:
    """
    Calculate weighted circuit depth where swap gates have a weight of swap_weight (default 3).
    Other gates have a weight of 1. This is the critical-path depth: for each gate we take
    max(current depth of involved qubits) + gate_weight, then update those qubits. Swap weight 3
    matches the depth cost of one swap when decomposed into 3 serial CX gates.
    circuit.data is assumed to be in execution order (Qiskit default).

    Args:
        circuit: QuantumCircuit to analyze
        swap_weight: Weight for swap gates (default 3)

    Returns:
        Weighted depth of the circuit
    """
    if len(circuit.qubits) == 0:
        return 0
    
    # Track the current depth for each qubit
    qubit_depths = {qubit: 0 for qubit in circuit.qubits}
    
    # Process each instruction in the circuit
    for instr, qargs, cargs in circuit.data:
        if len(qargs) == 0:
            continue
        
        # Determine the weight of this gate
        if instr.name == 'swap':
            gate_weight = swap_weight
        else:
            gate_weight = 1
        
        # For multi-qubit gates, find the maximum current depth among all involved qubits
        max_current_depth = max(qubit_depths[q] for q in qargs)
        
        # Update depth for all qubits involved in this gate
        new_depth = max_current_depth + gate_weight
        for q in qargs:
            qubit_depths[q] = new_depth
    
    # Return the maximum depth across all qubits
    return int(max(qubit_depths.values()))


def _is_rz_3pi_over_4(instr) -> bool:
    """True if instruction is RZ(±3π/4)."""
    angle = _rz_angle(instr)
    if angle is None:
        return False
    return isclose(angle, 3 * pi / 4, abs_tol=_T_ANGLE_TOL) or isclose(angle, -3 * pi / 4, abs_tol=_T_ANGLE_TOL)


def get_weighted_depth_rz34pi_as_2(circuit: QuantumCircuit, swap_weight: int = 3) -> int:
    """
    Weighted circuit depth with the same rules as get_weighted_depth, except:
    RZ(3π/4) and RZ(-3π/4) gates are assigned depth weight 2 instead of 1.
    (Swap weight unchanged; all other gates weight 1.)
    """
    if len(circuit.qubits) == 0:
        return 0

    qubit_depths = {qubit: 0 for qubit in circuit.qubits}

    for instr, qargs, cargs in circuit.data:
        if len(qargs) == 0:
            continue

        if instr.name == "swap":
            gate_weight = swap_weight
        elif _is_rz_3pi_over_4(instr):
            gate_weight = 2
        else:
            gate_weight = 1

        max_current_depth = max(qubit_depths[q] for q in qargs)
        new_depth = max_current_depth + gate_weight
        for q in qargs:
            qubit_depths[q] = new_depth

    return int(max(qubit_depths.values()))


def get_total_gate_count(circuit: QuantumCircuit) -> int:
    """Total gate count with no weighting; sum of all gate counts from count_ops()."""
    return sum(circuit.count_ops().values())


def get_total_gates_rz34pi_as_2(circuit: QuantumCircuit) -> int:
    """
    Total gate count where each gate counts 1, except RZ(3π/4) and RZ(-3π/4) count as 2.
    """
    total = 0
    for instr, qargs, cargs in circuit.data:
        if _is_rz_3pi_over_4(instr):
            total += 2
        else:
            total += 1
    return total


def read_qasm(file_path: str, name='Circuit', is_print=False) -> dict:
    """Load a QASM file and return its circuit info dict."""
    circuit = load_circuit_from_qasm(file_path)
    return get_circuit_info(circuit, name if name else file_path, is_print=is_print)


def read_quantum_circuit(circuit: QuantumCircuit, name='Circuit', is_print=False) -> dict:
    """Return the circuit info dict for an in-memory QuantumCircuit."""
    return get_circuit_info(circuit, name, is_print=is_print)

def get_circuit_info(circuit: QuantumCircuit, name, is_print=False) -> dict:
    qiskit_dict = circuit.count_ops()
    cx_count   = qiskit_dict.get('cx', 0)
    swap_count = qiskit_dict.get('swap', 0)
    h_count    = qiskit_dict.get('h', 0)
    rz_count   = qiskit_dict.get('rz', 0)
    x_count    = qiskit_dict.get('x', 0)

    total_gates = sum(qiskit_dict.values()) + 2 * swap_count
    cx_swap = cx_count + 3 * swap_count

    circuit_info = {
        'qiskit_info': qiskit_dict,
        'gates(always weighted)': total_gates,
        'weighted_cx' : cx_swap,
        'rz_gate': rz_count,
        't_gate': get_t_gate_count(circuit),
        'h_gate': h_count,
        'x_gate': x_count,
        'just_cx': cx_count,
        'swap': swap_count,
        'depth': circuit.depth(),
        'weighted_depth': get_weighted_depth(circuit, swap_weight=3),
    }

    if is_print:
        print(f'The details of the circuit: {name}')
        print(f'Qiskit info: {circuit_info["qiskit_info"]}')
        print(f'Gates: {circuit_info["gates(always weighted)"]}')
        print(f'Weighted CX: {circuit_info["weighted_cx"]}')
        print(f'RZ Gates: {circuit_info["rz_gate"]}')
        print(f'T Gates: {circuit_info["t_gate"]}')
        print(f'H Gates: {circuit_info["h_gate"]}')
        print(f'X Gates: {circuit_info["x_gate"]}')
        print(f'Just CX: {circuit_info["just_cx"]}')
        print(f'Swap Gates: {circuit_info["swap"]}')
        print(f'Depth: {circuit_info["depth"]}')
        print(f'Weighted Depth (swap=3): {circuit_info["weighted_depth"]}')

    return circuit_info

def print_circuit_graph(circuit, name='Circuit', output='text', output_path='output'):
    print('\n' + name)
    if output == 'text': print(circuit.draw())
    if output == 'mpl':  
        fig = circuit.draw(output='mpl')
        fig.savefig(output_path + '/' + name + '.png')

def decompose_circuit(circuit: QuantumCircuit, base_gate_sets = ['x', 'cx', 'h', 'rz']) -> QuantumCircuit:
    transpiled_circuit = transpile(circuit, basis_gates=base_gate_sets, optimization_level=0)
    return transpiled_circuit

def float_to_fraction_pi(angle, tol = 1e-8) -> Tuple[int, int]:
    # Convert a float number to a fraction of pi
    
    if isinstance(angle, sp.Basic):
        angle = float(angle.evalf()) 
    
    if not isinstance(angle, (float, int)):
        raise TypeError(f"Angle should be a float or int, got {type(angle)}")

    frac = Fraction(float(angle) / np.pi).limit_denominator(1000)
    a, b = frac.numerator, frac.denominator
    
    if abs(a/b * np.pi - angle) < tol:
        return a, b
    else:
        return None, None

def replace_angles(qasm_str) -> str:
    def replace_match(match):
        angle = float(match.group(1))
        a, b = float_to_fraction_pi(angle)
        if a is not None:
            if b==1:
                return f'rz({a}*pi) {match.group(2)};' if a != 1 else f'rz(pi) {match.group(2)};'
            else:
                return f'rz({a}*pi/{b}) {match.group(2)};' if a != 1 else f'rz(pi/{b}) {match.group(2)};'
        else:
            return match.group(0)
    
    new_qasm_str = re.sub(r'rz\(([\d\.\+\-eE]+)\)\s+([^\s;]+);', replace_match, qasm_str)
    return new_qasm_str

class QASMReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.circuit = QuantumCircuit.from_qasm_file(file_path)
        
    def get_info(self):
        return get_circuit_info(self.circuit, self.file_path)
        
    def print_info(self, name='Circuit'):
        get_circuit_info(self.circuit, name)
        print_circuit_graph(self.circuit, name)
    
    def change_angles_from_float_to_fraction_pi(self):
        for instr, qargs, cargs in self.circuit.data:
            if instr.name == 'rz':
                angle = instr.params[0]
                a, b = float_to_fraction_pi(angle)
                if a is not None:
                    if b == 1:
                        instr.params[0] = sp.pi * a
                    else:
                        instr.params[0] = sp.pi * sp.Rational(a, b)
                else:
                    print(f"Angle {angle} cannot be represented as a fraction of pi.")

    
    def decompose(self, base_gate_sets = ['x', 'cx', 'h', 'rz']):
        self.circuit = decompose_circuit(self.circuit, base_gate_sets)
        
    def output_qasm(self, file_path):
        with open(file_path, 'w') as file:
            # write circuit as qasm
            file.write(qasm2.dumps(self.circuit))
        print(f"Circuit saved to {file_path}")
        
    def output_mpl_circuits(self, name, output_path):
        if not os.path.exists(output_path):
                os.makedirs(output_path)
        file_name = self.file_path.split('/')[-1].split('.')[0]
        name = name if name else file_name
        
        try:
            print_circuit_graph(self.circuit, name=name, output='mpl', output_path=output_path)
        except Exception as e:
            print(f"Error in drawing circuit: {e}")

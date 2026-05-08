# src.sabre_run

from src.quantum_layouts import make_grid_layout
import numpy as np
import os, sys, time
from qiskit import transpile, QuantumCircuit
from qiskit.transpiler import CouplingMap
import qiskit.qasm2
from qiskit.providers.basic_provider import BasicProvider
from typing import Literal, Union
import networkx as nx

def sabre_run(input_path, output_path, num_qubits, coupling_map: Union[CouplingMap, nx.Graph, None]=None, auto_square:bool=False, layout_method:Literal['sabre', 'trivial']='sabre', optimization_level:int=2):
    qc = QuantumCircuit.from_qasm_file(input_path)
    print("Using qiskit.transpile to optimize circuits")
    backend = BasicProvider().get_backend('basic_simulator')

    if auto_square:
        side = int(np.floor(np.sqrt(qc.num_qubits)) + 1)
        sqrt_graph = make_grid_layout(side, side)
        input_coupling_map = CouplingMap(couplinglist=list(sqrt_graph.edges()))
    elif coupling_map is not None:
        # Support both CouplingMap and networkx Graph
        if isinstance(coupling_map, nx.Graph):
            input_coupling_map = CouplingMap(couplinglist=list(coupling_map.edges()))
        else:
            input_coupling_map = coupling_map
    else:
        raise ValueError("Either auto_square must be True or coupling_map must be provided")
        
    sabre_circuit = transpile(qc,
        coupling_map=input_coupling_map,
        backend=backend,
        layout_method=layout_method,
        routing_method='sabre',
        seed_transpiler=42, # Same seed for reproducibility
        basis_gates=['swap', 'cx', 'rz', 'h', 'x'],
        optimization_level=optimization_level)
    
    output_folder = os.path.dirname(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_path, 'w') as f:
        f.write(qiskit.qasm2.dumps(sabre_circuit))
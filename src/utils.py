import math, datetime

methods = [("row_heap", [10], [1, 3])] #,5,8,10

def log_message(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        f.write(f"{datetime.datetime.now()} - {message}\n")

class NoResultsError(Exception):
    pass

def get_all_methods():
    methods_list = []
    for m in methods:
        if m[0] != "row_heap":
            methods_list.append((m, m))
            continue
        for heap_size,group_size in [(h,g) for g in m[2] for h in m[1]]:
            if group_size == 1:
                methods_list.append((f"row_heap_{heap_size}", "row_heap", heap_size, 1))
            else:
                methods_list.append((f"group_{group_size}_row_heap_{heap_size}", "row_heap", heap_size, group_size))
    return methods_list

class PiAngle():
    def __init__(this, piMult:float):
        this.mult = piMult
    
    def get_normalized_mult(this):
        out = this.mult
        while out > 1:
            out -= 2
        while out <= -1:
            out += 2
        return out

    def __add__(this, other):
        if other == 0:
            return this
        if isinstance(other, PiAngle):
            out = PiAngle(this.mult + other.mult)
            out.mult = out.get_normalized_mult()
            return out
        return (math.pi * this.mult) + other
    
    def __radd__(this, other):
        return this.__add__(other)

    def __neg__(this):
        return PiAngle(-this.mult)

    def __sub__(this, other):
        return this + (-other)

    def __rsub__(this, other):
        return other + (-this)

    def __mul__(this, other):
        return PiAngle(this.mult * other)

    def __rmul__(this, other):
        return PiAngle(other * this.mult)

    def __truediv__(this, other):
        return PiAngle(this.mult / other)

    def __rtruediv__(this, other):
        return other / (this.mult * math.pi)

    def __eq__(this, other):
        if isinstance(other, PiAngle):
            return this.get_normalized_mult() == other.get_normalized_mult()
        return this.get_normalized_mult() * math.pi == other

    def __neq__(this, other):
        return not (this == other)
    
    def __repr__(this) -> str:
        return f"{this.mult}*pi"
    
    def __float__(this):
        return this.mult * math.pi
    
    def __str__(this):
        return this.__repr__()

if __name__ == '__main__':
    #testing, please ignore
    p = PiAngle(0.5)
    q = PiAngle(-0.2)
    r = PiAngle(1)
    print(p+q)
    print(p*0.8+5-q*0.56)
    print(p*2, p*3, p*4, p*5)
    print(p/6)
    print(-q + p*4 + q == 0)
    print(-r / 4, (-r+PiAngle(1)-PiAngle(1)) / 4)
    print((3*r) / 4, (3*r+PiAngle(1)-PiAngle(1)) / 4)

def is_in_testing_circuit_names(input_qasm_files, testing_circuit_names)->dict[str, str]:
    """
    Check if input QASM files are in the testing circuit names list.
    
    Args:
        input_qasm_files: List of QASM filename or single QASM filename
        testing_circuit_names: List of circuit names to test against
        
    Returns:
        dict[str, str]: Dictionary with circuit names as keys and corresponding QASM filename as values
        
    Note:
        This function handles filenames with parameters (e.g., 'ss(1)') by only matching the base filename before parentheses.
    """
    # Convert single filename to list for uniform processing
    if isinstance(input_qasm_files, str):
        input_qasm_files = [input_qasm_files]
    
    result = {}
    
    for qasm_filename in input_qasm_files:
        # Extract base filename without extension and parameters
        base_name = qasm_filename
        
        # Remove .qasm extension if present
        if base_name.endswith('.qasm'):
            base_name = base_name[:-5]
        
        # Remove parameters in parentheses if present (e.g., 'ss(1)' -> 'ss')
        if '(' in base_name:
            base_name = base_name.split('(')[0]
        
        # Remove 'optimized_nam_' prefix if present, this prefix is added by the QUESO benchmark
        if "optimized_nam_" == base_name[:14]:
            base_name = base_name[14:]
        if "_quartzOutput" == base_name[-13:]:
            base_name = base_name[:-13]
        
        # Check if the base name is in the testing circuit names
        if base_name in testing_circuit_names:
            result[base_name] = qasm_filename
    
    return result

class Mapping():
    def __init__(this):
        this.log_qubits:list[int] = []
        this.phys_qubits:list[int] = []

    def map_qubit(this, logical_qubit:int, physical_qubit:int):
        if logical_qubit in this.log_qubits:
            raise Exception("Illegal Mapping operation")
        if physical_qubit in this.phys_qubits:
            raise Exception("Illegal Mapping operation")
        this.log_qubits.append(logical_qubit)
        this.phys_qubits.append(physical_qubit)

    def unmap_logical_qubit(this, logical_qubit:int):
        if logical_qubit not in this.log_qubits:
            raise Exception("Illegal Mapping operation")
        index = this.log_qubits.index(logical_qubit)
        this.log_qubits.pop(index)
        this.phys_qubits.pop(index)

    def get_logical_from_phys(this, physical_qubit:int):
        try:
            index = this.phys_qubits.index(physical_qubit)
            return this.log_qubits[index]
        except ValueError:
            return None
        
    def get_physical_from_log(this, logical_qubit:int):
        try:
            index = this.log_qubits.index(logical_qubit)
            return this.phys_qubits[index]
        except ValueError:
            return None

    def swap_physical_qubits(this, phys_qubit_1:int, phys_qubit_2:int):
        phys1_mapped = this.get_logical_from_phys(phys_qubit_1) != None
        phys2_mapped = this.get_logical_from_phys(phys_qubit_2) != None

        if phys1_mapped == True:
            index1 = this.phys_qubits.index(phys_qubit_1)
        if phys2_mapped == True:
            index2 = this.phys_qubits.index(phys_qubit_2)
        if phys1_mapped == True:
            this.phys_qubits[index1] = phys_qubit_2
        if phys2_mapped == True:
            this.phys_qubits[index2] = phys_qubit_1

    def map_physical_qubit_at_logical(this, logical_qubit:int, physical_qubit:int):
        if logical_qubit not in this.log_qubits:
            raise Exception()
        this.phys_qubits[this.log_qubits.index(logical_qubit)] = physical_qubit

    def copy(this):
        out = Mapping()
        out.log_qubits = this.log_qubits.copy()
        out.phys_qubits = this.phys_qubits.copy()
        return out
    
    def __eq__(this, value: object) -> bool:
        if value.__class__ != Mapping:
            return False
        value:Mapping = value
        this_pairs = list(sorted(zip(this.log_qubits, this.phys_qubits)))
        val_pairs = list(sorted(zip(value.log_qubits, value.phys_qubits)))

        return this_pairs == val_pairs 
    
    def __repr__(this) -> str:
        return f'mapping: {list(zip(this.log_qubits, this.phys_qubits))}'


# ---------------------------------------------------------------------------
# Phase-polynomial block statistics (moved from phasepoly_reader.py)
# ---------------------------------------------------------------------------

def get_phasepoly_block_stats(circ, wide_mode=False):
    """
    Get the phasepoly block stats from a circuit.
    Returns (full_circuit_stats, block_stats_list).

    Each block stat dict has the form:
      {"index": i, "size": ..., "qubits_count": ..., "cx_count": ...,
       "rz_count": ..., "h_count": ..., "x_count": ..., "depth": ...}
    """
    stats = circ.get_stats()
    stats = {'index': 0, **stats}
    if wide_mode:
        block_stats = circ.partition_to_phasePoly_wide(size_metric='rotations', maxsize=None)
    else:
        block_stats = circ.partition_to_phasePoly(size_metric='rotations', maxsize=None)

    for i, block_stat in enumerate(block_stats, start=1):
        new_block_stat = {"index": i}
        new_block_stat.update(block_stat)
        block_stat.clear()
        block_stat.update(new_block_stat)

    return stats, block_stats
import src.circuits as circuits
from typing import Any
from collections import deque
import math, src.utils as utils

def read_qasm(filename:str):
    out:circuits.Circuit
    with open(filename, 'r') as fp:
        for line in fp:
            comment_index = line.find(r"//")
            if comment_index != -1:
                line = line[:comment_index]
            comment_index = line.find(r"#")
            if comment_index != -1:
                line = line[:comment_index]
            
            if line.strip() == "":
                continue

            command, qbits = line.strip().strip(";").split(" ", 1)

            if command == "phasepoly":
                out.append_node(__generate_phasepoly(qbits))
                continue

            cbits = None
            if qbits.find("->") != -1:
                qbits, cbits = qbits.split("->")


            if command == "include":
                continue
            if command == "OPENQASM":
                continue

            params = []
            if "(" in command:
                left_paren = command.index("(")
                right_paren = command.index(")")

                params = command[left_paren+1:right_paren].split(",")
                params = [math_str_to_float(p) for p in params]
                command = command[:left_paren]

            qbit_strs = qbits.strip().split(',')
            qbit_vals = []
            for qs in qbit_strs:
                qbit_vals.append(int(qs.strip().strip('q[]')))

            if cbits != None:
                cbit_strs = cbits.strip().split(',')
                cbit_vals = []
                for cs in cbit_strs:
                    cbit_vals.append(int(cs.strip().strip('c[]')))

            if command == "qreg":
                out = circuits.Circuit(qbit_vals[0])
                continue
            out.append_node(circuits.Gate(command, qbit_vals, params))
    return out

def __generate_phasepoly(input:str):
    qbits, rotations, out_qbits = input.split("->")
    
    qbit_strs = qbits.strip().split(',')
    qbit_vals = [int(qs.strip().strip('q[]')) for qs in qbit_strs]

    rotation_strs = rotations.strip().split(',')
    #print(rotation_strs)
    if len(rotation_strs) > 0 and rotation_strs[0] == "":
        rotation_strs = []
    rotation_vals = []
    for rs in rotation_strs:
        angle, parity = rs.strip().strip(')').split('(')
        angle_val = math_str_to_float(angle.strip())
        parity_qbit_strs = parity.strip().split('*')
        #print(parity_qbit_strs)
        parity_qbit_vals = set(int(qs.strip().strip('q[]')) for qs in parity_qbit_strs)

        rotation_vals.append((angle_val,parity_qbit_vals))

    out_qbit_strs = out_qbits.strip().split(',')
    out_qbit_str_lists = [qs.strip().split('*') for qs in out_qbit_strs]
    out_qubit_vals:dict[int,set[int]] = dict()
    for p, parity in zip(qbit_vals, out_qbit_str_lists):
        if len(parity) > 0 and parity[0] == "_":
            continue
        out_parity = set(int(qs.strip().strip('q[]')) for qs in parity)
        out_qubit_vals[p] = out_parity
    
    return circuits.PhasePoly(qbit_vals, rotation_vals, out_qubit_vals)

def write_qasm(circ:circuits.Circuit, filename:str):
    with open(filename, 'w') as fp:
        fp.write('OPENQASM 2.0;\ninclude "qelib1.inc";\n')
        circ_size = max(circ.qubits)+1
        fp.write(f"qreg q[{circ_size}];\n")
        for node in circ.get_sequence():
            fp.write(node.to_instr()+"\n")


def math_str_to_float(math_str):
    try:
        out = float(math_str)
        return out
    except ValueError:
        pass
    #Below is WIP
    tokens:deque[tuple[str,Any]] = deque()
    ptr = 0
    while ptr < len(math_str):
        c:str = math_str[ptr]
        if c in "-*/":
            tokens.append((c, None))
        elif c.isspace():
            continue
        else:
            if c == "p":
                if len(math_str) - ptr >= 2 and math_str[ptr:ptr+2] == "pi":
                    tokens.append(("num", utils.PiAngle(1)))
                    ptr += 1
                else:
                    raise Exception("Parsing Error")
            elif c in "1234567890.":
                start_ptr = ptr
                while ptr < len(math_str)-1 and math_str[ptr+1] in "1234567890.":
                    ptr += 1
                tokens.append(("num", float(math_str[start_ptr:ptr+1])))
            else:
                raise Exception("Parsing Error")
            if len(tokens) > 1 and tokens[-2][0] == "-":
                _, num = tokens.pop()
                tokens.pop()
                tokens.append(("num", -num))
            
        ptr += 1
    #print(tokens)

    while len(tokens) > 1:
        if len(tokens) == 2 or len(tokens) <= 0:
            raise Exception(f"Parsing Error: {tokens}")
        if tokens[0][0] == tokens[2][0] == "num":
            _, num1 = tokens.popleft()
            op, _ = tokens.popleft()
            _, num2 = tokens.popleft()
            if op == "*":
                tokens.appendleft(("num",num1*num2))
            elif op == "/":
                tokens.appendleft(("num",num1/num2))
            else:
                raise Exception(f"Parsing Error: {tokens}")
            continue
        raise Exception(f"Parsing Error: {tokens}")
    
    return tokens[0][1]

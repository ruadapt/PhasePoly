from __future__ import annotations
from typing import Callable, Mapping, Optional, Union
from typing_extensions import Self, Any, Literal #For annotations only
import math
import src.utils as utils
from collections import deque, defaultdict
import copy

import networkx as nx

#class for graph nodes. Used to represent single gates and phase polynomials
#notes for use in comparisons: 
# * Node.nodeType compares types of nodes, ignoring parameters and additional data
# * Node.is_equivalent_to() compares if 2 nodes have equivalent data, including or excluding ins/outs based on a given parameter
# * == operator behavior is python default

class Node:
    #note that ins and outs need to be initialized before proper use, see connect_to() and connect_from()
    sig = 0 #<--- debug purposes only
    
    def __init__(this, qubits:list[int], ins:(dict[int,Node]|None) = None, outs:(dict[int,Node]|None) = None):
        this.qubits = qubits.copy()
        this.nodeType = "generic"
        this.sig = Node.sig
        Node.sig += 1
        if ins != None:
            this.ins = ins.copy()
        else:
            this.ins:dict[int,Node] = dict()
        
        if outs != None:
            this.outs = outs.copy()
        else:
            this.outs:dict[int,Node] = dict()

    #makes edge directing this node to 1 other node
    def connect_to(this, target_node:Node, qubit:int):
        if this.outs.get(qubit) != None:
            this.outs[qubit].ins.pop(qubit)
        if target_node == None:
            return
        if target_node.ins.get(qubit) != None:
            target_node.ins[qubit].outs.pop(qubit)
        this.outs[qubit] = target_node
        target_node.ins[qubit] = this

    def connect_from(this, source_node:Node, qubit:int):
        if this.ins.get(qubit) != None:
            this.ins[qubit].outs.pop(qubit)
        if source_node == None:
            return
        if source_node.outs.get(qubit) != None:
            source_node.outs[qubit].ins.pop(qubit)
        this.ins[qubit] = source_node
        source_node.outs[qubit] = this

    #find previous node applying to a qubit
    def prev(this, qubit:int):
        out = this.ins.get(qubit)
        '''if out == None:
            raise Exception("No previous node exists")'''
        return out
    
    #find next node applying to a qubit
    def next(this, qubit:int):
        out = this.outs.get(qubit)
        '''if out == None:
            raise Exception("No next node exists")'''
        return out
    
    #attach this node after a set of previous nodes, while remaking the graph edges correctly
    def attach_after(this, prev_nodes:dict[int,Node]):
        for qubit, node in prev_nodes.items():
            this.connect_to(node.outs[qubit], qubit)
            this.connect_from(node, qubit)

    #attach this node before a set of next nodes, while remaking the graph edges correctly
    #not used much
    def attach_before(this, next_nodes:dict[int,Node]):
        for qubit, node in next_nodes.items():
            this.connect_from(node.ins[qubit], qubit)
            this.connect_to(node, qubit)

    #remove this node from its current circuit, while remaking the graph edges correctly
    def remove_from_circuit(this):
        for q in set(this.ins.keys()) & set(this.outs.keys()):
            this.prev(q).connect_to(this.next(q), q)
        for q, node in this.ins.items():
            del node.outs[q]
        for q, node in this.outs.items():
            del node.ins[q]
        this.ins = dict()
        this.outs = dict()

    def shift_forward(this, qubit:int):
        next = this.next(qubit)
        nnext = next.next(qubit)
        prev = this.prev(qubit)
        
        next.connect_to(this, qubit)
        next.connect_from(prev, qubit)
        this.connect_to(nnext, qubit)

    def shift_backward(this, qubit:int):
        prev = this.prev(qubit)
        pprev = prev.prev(qubit)
        next = this.next(qubit)
        
        prev.connect_to(next, qubit)
        this.connect_from(pprev, qubit)
        this.connect_to(prev, qubit)

    def to_circuit(this):
        return Circuit(ins=this.ins, outs=this.outs)

    def to_instr(this):
        return "<?>"
    
    def __repr__(this) -> str:
        return this.to_instr() + f" #{this.sig}"
    
    #checks if this and the other gate is equivalent, ignoring input/outputs nodes. 
    #should not be used to replace ==/__eq__, as that should be reserved for default object compare behavior
    def is_equivalent_to(this, other_node:Node, ignore_qubits:bool = False):
        if this.__class__ != other_node.__class__:
            return False
        return this.nodeType == other_node.nodeType and (ignore_qubits or this.qubits == other_node.qubits)
    
    def copy_disconnected(this):
        return Node(this.qubits)
    
#subclass that represents gates
class Gate(Node):
    def __init__(this, gateType:str, qubits:list[int], params:list[float|utils.PiAngle] = []):
        super().__init__(qubits)
        this.nodeType = gateType
        this.params = params.copy()

    #returns readable qasm output
    def to_instr(this):
        out = this.nodeType
        if this.params != None and len(this.params) > 0:
            out += f"({','.join([str(p) for p in this.params])})"
        out += " "+",".join(f"q[{q}]" for q in this.qubits)
        return out + ';'
    
    def copy_disconnected(this):
        return Gate(this.nodeType, this.qubits, this.params)
    
    def is_equivalent_to(this, other_node:Node, ignore_qubits:bool = False):
        if not super().is_equivalent_to(other_node, ignore_qubits=ignore_qubits):
            return False
        if not isinstance(other_node, Gate):
            return False
        return this.params == other_node.params

class CommentLine(Node):
    def __init__(this, qubits:list[int], message:str = ""):
        super().__init__(qubits)
        this.message = message
        this.nodeType = "commentLine"

    def to_instr(this):
        return '//'+this.message
    
    def copy_disconnected(this):
        return CommentLine(this.qubits, this.message)
    
    def is_equivalent_to(this, other_node:Node, ignore_qubits:bool = False):
        if not super().is_equivalent_to(other_node, ignore_qubits=ignore_qubits):
            return False
        if not isinstance(other_node, CommentLine):
            return False
        return this.message == other_node.message

#subclass that represents a phase polynomial
class PhasePoly(Node):
    def __init__(this, qubits:list[int], rotations:list[tuple[float|utils.PiAngle,set[int]]], affineOut:dict[int,set[int]]):
        super().__init__(qubits)
        this.rotations = rotations.copy()
        this.affineOut = affineOut.copy()
        this.nodeType = "phasePoly"
        this.possible_circuits:list[Circuit] = list()
        pass

    def remap_inputs_change_parities(this, mapping:dict[int,set[int]]):
        new_qubits = this.qubits.copy()
        out = PhasePoly(list(new_qubits), copy.deepcopy(this.rotations), copy.deepcopy(this.affineOut))
        for q in mapping.keys():
            for i,rot in enumerate(this.rotations):
                if q not in rot[1]:
                    continue
                new_rot_parity = out.rotations[i][1] ^ {q} ^ mapping[q]
                out.rotations[i] = (rot[0], new_rot_parity)
            for i,affineOut in this.affineOut.items():
                if q not in affineOut:
                    continue
                out.affineOut[i] ^= {q} ^ mapping[q]
        return out

    def remap_inputs_preserve_parities(this, mapping:dict[int,set[int]]):
        out = PhasePoly(this.qubits.copy(), copy.deepcopy(this.rotations), copy.deepcopy(this.affineOut))

        new_mapping = copy.deepcopy(mapping)
        for j in [j for i in new_mapping.values() for j in i]:
            if j not in new_mapping.keys():
                raise Exception("mapping not valid. (parities should form a singular matrix)")
        checked_qs = set()
        for q in new_mapping.keys():
            checked_qs.add(q)
            new_parity = new_mapping[q].copy()
            if q not in new_parity:
                swap_target = [t for t in new_parity if t not in checked_qs][0]
                for i,rot in enumerate(out.rotations):
                    if swap_target not in rot[1]:
                        continue
                    new_rot_parity = out.rotations[i][1] ^ {q}
                    out.rotations[i] = (rot[0], new_rot_parity)
                for i,affineOut in out.affineOut.items():
                    if swap_target not in affineOut:
                        continue
                    out.affineOut[i] ^= {q}
                for i,parity in new_mapping.items():
                    if swap_target not in parity:
                        continue
                    new_mapping[i] ^= {q}
            new_parity = new_mapping[q].copy()
            for i,rot in enumerate(out.rotations):
                if q not in rot[1]:
                    continue
                new_rot_parity = out.rotations[i][1] ^ {q} ^ new_parity
                out.rotations[i] = (rot[0], new_rot_parity)
            for i,affineOut in out.affineOut.items():
                if q not in affineOut:
                    continue
                out.affineOut[i] ^= {q} ^ new_parity
            for i,parity in new_mapping.items():
                if q not in parity:
                    continue
                new_mapping[i] ^= {q} ^ new_parity

        return out
    
    def replace_qubit(this, old_qubit, new_qubit):
        if old_qubit not in this.qubits:
            raise Exception(f"Cannot replace quibit that doesn't exist : {old_qubit}")
        out = this.copy_disconnected()
        
        replace_index = out.qubits.index(old_qubit)
        out.qubits[replace_index] = new_qubit

        for i,rot in enumerate(this.rotations):
            if old_qubit not in rot[1]:
                continue
            new_rot_parity = this.rotations[i][1] ^ {old_qubit} ^ {new_qubit}
            out.rotations[i] = (rot[0], new_rot_parity)
        for i,affineOut in this.affineOut.items():
            if old_qubit not in affineOut:
                continue
            out.affineOut[i] ^= {old_qubit} ^ {new_qubit}
        out.affineOut[new_qubit] = out.affineOut.pop(old_qubit)
        '''for circ in out.possible_circuits:
            circ.remap(lambda x: new_qubit if x == old_qubit else x)'''
        out.possible_circuits = []
        
        return out

    def to_instr(this):
        out = "phasepoly " + ",".join([f"q[{i}]" for i in this.qubits])+" -> "

        rotations = sorted(this.rotations, key=lambda x:list(sorted(x[1])))
        out += ",".join([str(t)+"("+"*".join([f"q[{i}]" for i in q])+")" for t,q in rotations]) + " -> "
        out += ",".join(["*".join([f"q[{i}]" for i in this.affineOut[q]]) if q in this.affineOut.keys() else "_" for q in this.qubits])
        return out + ';'
    
    def copy_disconnected(this):
        out = PhasePoly(this.qubits, copy.deepcopy(this.rotations), copy.deepcopy(this.affineOut))
        out.possible_circuits = this.possible_circuits.copy()
        return out
    
    def is_equivalent_to(this, other_node:Node, ignore_qubits:bool = False):
        if not super().is_equivalent_to(other_node, ignore_qubits = ignore_qubits):
            return False
        if not isinstance(other_node, PhasePoly):
            return False
        return all(this.affineOut[q] == other_node.affineOut[q] for q in this.qubits) and sorted(this.rotations, key=lambda x: (sorted(list(x[1])),x[0])) == sorted(other_node.rotations, key=lambda x: (sorted(list(x[1])),x[0]))

phasepoly_legal_gates = {"cx", "cnot", "x", "not", "rz", "swap", "i"}

#actual circuit class. Should work as a graph.
class Circuit:
    # For input params, its either (size:int) or (qubits:list[int]) for a new circuit, or (ins:dict[int,Node], outs,dict[int,Node]) for a subcircuit. 
    
    # Note: It is suggested against making multiple subcircuits on the same circuit due to how the ins/outs system works.
    # Any alternate solutions to define circuits in ways other than exclusive ins/outs are welcome, but remember that this should still support empty circuits.
    def __init__(this, size:(int|None) = None, qubits:(list[int]|None) = None, ins:(dict[int,Node]|None) = None, outs:(dict[int,Node]|None) = None):
        if ins != None and outs != None:
            if set(ins.keys()) != set(outs.keys()):
                raise Exception("Invalid")
            this.qubits = sorted(list(ins.keys()))
            this.ins = ins.copy()
            this.outs = outs.copy()
            this.is_subcircuit = True #temp measure
            #this.get_sequence()
            return
        if (size == None) == (qubits == None):
            raise Exception("Invalid")
        if qubits != None:
            this.qubits = qubits
        elif size != None:
            this.qubits = list(range(size))
        this.ins:dict[int,Node] = dict()
        this.outs:dict[int,Node] = dict()
        this.is_subcircuit = False
        for q in this.qubits:
            this.ins[q] = Node([q])
            this.outs[q] = Node([q])
            this.ins[q].connect_to(this.outs[q], q)

    #returns all nodes in the circuit as a list of nodes, ordered with dependency in mind.
    def get_sequence(this):
        sequence:list[Node] = []
        tracked = set()
        buffer:deque[Node] = deque()
        for q, in_node in this.ins.items():
            tracked.add(in_node)
            node = in_node.next(q)
            if in_node not in node.ins.values():
                raise Exception(f"Improper Graph Structure: {this.qubits}, {this.ins} -> {this.outs}")
            if node in this.ins.values():
                raise Exception(f"Improper Graph Structure: {this.qubits}, {this.ins} -> {this.outs}")
            if node not in buffer and node not in tracked:
                buffer.append(node)

        skip_streak = 0
        #print("\nget_seq buffer:")
        while len(buffer) > skip_streak:
            #print(buffer)
            target = buffer.popleft()
            if not all(prev_node in tracked for prev_node in target.ins.values()) and target not in this.outs.values():
                skip_streak += 1
                buffer.append(target)
                continue
            skip_streak = 0
            if target not in this.outs.values():
                for node in target.outs.values():
                    if target not in node.ins.values():
                        raise Exception(f"Improper Graph Structure: {this.qubits}, {this.ins} -> {this.outs}")
                    if node not in buffer and node not in tracked:
                        buffer.append(node)
                sequence.append(target)
            tracked.add(target)

        if len(buffer) != 0:
            raise Exception(f"Improper Graph Structure: {this.qubits}, {this.ins} -> {this.outs}\n buffer:{buffer}")
        return sequence

    def append_node(this, new_node:Node):
        if any(q not in this.qubits for q in new_node.qubits):
            raise Exception(f"Invalid Qubit input: {list(q for q in new_node.qubits if q not in this.qubits)}")
        
        prev_nodes:dict[int,Node] = dict()
        for q in new_node.qubits:
            prev_nodes[q] = this.outs[q].prev(q)
        new_node.attach_after(prev_nodes)

    def prepend_node(this, new_node:Node):
        if any(q not in this.qubits for q in new_node.qubits):
            raise Exception(f"Invalid Qubit input")
        
        next_nodes:dict[int,Node] = dict()
        for q in new_node.qubits:
            next_nodes[q] = this.ins[q].next(q)
        new_node.attach_before(next_nodes)

    def append_circuit(this, new_circuit:Self):
        if any(q not in this.qubits for q in new_circuit.qubits):
            raise Exception("Invalid Qubit input")
        
        for q,n in new_circuit.ins.items():
            this.outs[q].prev(q).connect_to(n.next(q), q)
        for q,n in new_circuit.outs.items():
            n.prev(q).connect_to(this.outs[q], q)

    
    #remaps the circuits qubits (very jank)
    #DO NOT USE WITH PHASEPOLYS 
    #In particlur, the function input parameter is very awkward to use. See remap_demo.py
    def remap(this, mapping):
        if this.is_subcircuit:
            raise Exception("Illegal Function") #yes, this is horrible design. I'll refactor this eventually.
        
        def remap_list(in_list:list[int]):
            out_list:list[int] = []
            for num in in_list: 
                out_list.append(mapping(num))
            return out_list
        
        def remap_dict(in_dict:dict[int, Any]):
            out_dict:dict[int, Any] = dict()
            for i, node in in_dict.items():
                new_i = mapping(i)
                if new_i in out_dict.keys():
                    raise Exception("Mapping error: QuBit Overlap")
                out_dict[new_i] = node
            return out_dict

        this.qubits = remap_list(this.qubits)

        for in_node in this.ins.values():
            in_node.qubits = remap_list(in_node.qubits)
            in_node.outs = remap_dict(in_node.outs)
        this.ins = remap_dict(this.ins)
        
        sequence = this.get_sequence()
        for node in sequence:
            node.qubits = remap_list(node.qubits)
            node.ins = remap_dict(node.ins)
            node.outs = remap_dict(node.outs)
            if isinstance(node, PhasePoly):

                #Completely untested implementation. #TODO
                replace = node.remap_inputs_preserve_parities(dict([(q, {mapping(q)}) for q in node.qubits]))
                node.rotations = replace.rotations
                node.affineOut = replace.affineOut
                for c in node.possible_circuits:
                    c.remap(mapping)
                #raise Exception("PhasePolys not supported by remap at this time")
        
        for out_node in this.outs.values():
            out_node.qubits = remap_list(out_node.qubits)
            out_node.ins = remap_dict(out_node.ins)
        this.outs = remap_dict(this.outs)
        
        this.get_sequence() # this is to verify the circuit structure, for debug purposes

    #creates a copy, with different node objects
    def copy(this):
        out_circuit = Circuit(qubits=this.qubits)
        for node in this.get_sequence():
            out_circuit.append_node(node.copy_disconnected())
        return out_circuit
    
    #copy the target circuit to the given inputs/outputs. 
    #Note that the parameter circuit is the circuit to be replaced and must match the given circuit's qubits.
    #the circuit object the function is called at will be unchanged.
    def replace_at(this, subcircuit_to_replace:Self):
        if not (set(this.qubits) == set(subcircuit_to_replace.qubits)):
            raise Exception("Invalid Qubits for Parameters")
        sequence = this.get_sequence()
        for node in sequence:
            if not set(node.qubits).issubset(set(this.qubits)):
                raise Exception("Circuit to be placed is not independent")
        
        for q in subcircuit_to_replace.qubits:
            subcircuit_to_replace.ins[q].connect_to(subcircuit_to_replace.outs[q], q)
        for node in sequence:
            if not set(node.qubits).issubset(set(subcircuit_to_replace.qubits)):
                raise Exception("Circuit to replace is not independent")
            subcircuit_to_replace.append_node(node.copy_disconnected())

    #Checks to see if this and another circuit/subcircuit match gates. remap is an optional function to remap other_circuit
    #Unused
    def structural_match(this, other_circuit:Self, remap = None):
        if remap == None:
            remap = lambda x : x
        for q in this.qubits:
            c1ptr = this.ins[q].next(q)
            if remap(q) not in other_circuit.ins.keys():
                return False
            c2ptr = other_circuit.ins[remap(q)].next(remap(q))
            while True:
                if c1ptr == this.outs[q] and c2ptr == other_circuit.outs[remap(q)]:
                    break
                if c1ptr != this.outs[q] and c2ptr != other_circuit.outs[remap(q)]:
                    if not c1ptr.is_equivalent_to(c2ptr):
                        return False
                else:
                    return False
                c1ptr = c1ptr.next(q)
                c2ptr = c2ptr.next(remap(q))
        return True

    def depth(this, gate_latency: Union[Mapping[str, float], Callable[['Node'], float], None] = None, default_latency: float = 1.0):
        """
        Calculate the circuit depth with configurable gate latencies.

        Args:
            gate_latency: A mapping from gate type to latency, or a callable that
                accepts a Node and returns its latency. If None, every gate has
                latency 1.
            default_latency: Fallback latency when a gate type is missing in the
                mapping or when the callable returns a falsy value.
                
        Example_1:
            config = {
                "h": 1,
                "x": 1,
                "cx": 4,
                "t": 2,
                "tdg": 2
            }

            depth = circuit.depth(gate_latency=config)
            print(depth) 
            
        Example_2:
            def my_latency(node):
                if node.nodeType == "cx":
                    return 4.0
                if node.nodeType == "swap":
                    return 6.0
                if len(node.qubits) == 1:
                    return 1.0
                return None

            depth = circuit.depth(gate_latency=my_latency, default_latency=1.0)
            print(depth)
        """
        if len(this.qubits) == 0:
            return 0.0

        def _resolve_latency(node: Node) -> float:
            if callable(gate_latency):
                value = gate_latency(node)
                if value is None:
                    raise ValueError(f"Unrecognized gate latency for node type '{node.nodeType}' via callable.")
                return float(value)
            if isinstance(gate_latency, Mapping):
                if node.nodeType not in gate_latency:
                    raise ValueError(f"Unrecognized gate latency for node type '{node.nodeType}'.")
                return float(gate_latency[node.nodeType])
            return 1.0

        current_depths = [0.0] * len(this.qubits)
        for n in this.get_sequence():
            if len(n.qubits) == 0:
                continue
            current_indices = [this.qubits.index(q) for q in n.qubits]
            latency = _resolve_latency(n)
            start_depth = max(current_depths[i] for i in current_indices)
            updated_depth = start_depth + latency
            for i in current_indices:
                current_depths[i] = updated_depth

        return max(current_depths)
    
    def get_stats(this):
        stats = dict()
        seq = this.get_sequence()
        stats['size'] = len(seq)
        stats['qubits_count'] = len(this.qubits)
        stats['cx_count'] = len([True for g in seq if g.nodeType == 'cx'])
        stats['rz_count'] = len([True for g in seq if g.nodeType == 'rz'])
        stats['h_count'] = len([True for g in seq if g.nodeType == 'h'])
        stats['x_count'] = len([True for g in seq if g.nodeType == 'x'])
        stats['depth'] = this.depth()
        return stats
    
    def extend_quibits(this, new_qubit:int):
        if this.is_subcircuit:
            raise Exception("Subcircuits should not be extended with extend_qubits")
        this.ins[new_qubit] = Node([new_qubit])
        this.outs[new_qubit] = Node([new_qubit])
        this.ins[new_qubit].connect_to(this.outs[new_qubit], new_qubit)

    def partition_to_phasePoly(this, preserve_original=False, size_metric:Literal['qibit_rotation_product','qubits','rotations']="qibit_rotation_product", maxsize:int|None = None):
        stats = []
        tracked = set()

        for node in this.get_sequence():
            if node in tracked:
                continue
            tracked.add(node)
            if node.nodeType not in phasepoly_legal_gates:
                continue
            #print(f"$ {node}")

            seq_nums = dict((b,a) for a,b in enumerate(this.get_sequence()))

            rotation_count = 0

            current_ins:dict[int,Node] = node.ins.copy()
            current_outs:dict[int,Node] = node.outs.copy()
            pending:set[tuple[Node,bool]] = set()
            buffer = deque()
            for q in node.qubits:
                buffer.extend([(q, node.next(q), True), (q, node.prev(q), False)])

            def cost(qubits, rotations):
                if size_metric == 'qubits':
                    return qubits
                elif size_metric == 'rotations':
                    return rotations
                else:
                    return qubits*rotations

            def parse_anchor(qubit:int, node:Node, direction:bool):
                nonlocal rotation_count
                ptr = node
                if direction:
                    current_bound = current_outs
                else:
                    current_bound = current_ins
                while ptr != None and ptr.nodeType in phasepoly_legal_gates:
                    if ptr.nodeType in ("cnot", "cx", "swap"):
                        other_q = ptr.qubits[0] if ptr.qubits[0] != qubit else ptr.qubits[1] 
                        if other_q not in current_ins.keys():
                            '''if maxsize != None and cost(len(current_ins.keys())+1, rotation_count) > maxsize:
                                break'''
                            min_out_num = min(seq_nums.get(n,math.inf) for n in current_outs.values())
                            max_in_num = max(seq_nums.get(n,-1) for n in current_ins.values())

                            #print("min_out_num", min_out_num, "max_in_num", max_in_num, ptr, other_q)

                            #anti self-dependency measures

                            stop_flag = False
                            backbuffer = deque([ptr.prev(other_q)])
                            tracked_b = set()
                            while len(backbuffer) > 0:
                                #print("backbuffer", [f"({q},{seq_nums.get(q)})" for q in backbuffer])
                                item = backbuffer.popleft()
                                tracked_b.add(item)
                                if item == None:
                                    raise Exception("Out of bounds (this shouldn't happen)")
                                if item in current_outs.values():
                                    #print(f"back stopped at {item}")
                                    stop_flag = True
                                    break
                                if item in seq_nums.keys() and seq_nums[item] > min_out_num:
                                    for o in item.ins.values():
                                        if o not in tracked_b and o not in backbuffer:
                                            backbuffer.append(o)
                            if stop_flag:
                                pending.add((ptr,direction))
                                break
                            forwardbuffer = deque([ptr.next(other_q)])
                            tracked_f = set()
                            while len(forwardbuffer) > 0:
                                #print("forwardbuffer", [f"({q},{seq_nums.get(q)})" for q in forwardbuffer])
                                item = forwardbuffer.popleft()
                                tracked_f.add(item)
                                if item == None:
                                    raise Exception("Out of bounds (this shouldn't happen)")
                                if item in current_ins.values():
                                    #print(f"forward stopped at {item}")
                                    stop_flag = True
                                    break
                                if item in seq_nums.keys() and seq_nums[item] < max_in_num:
                                    for o in item.outs.values():
                                        if o not in tracked_f and (o not in forwardbuffer):
                                            forwardbuffer.append(o)
                            if stop_flag:
                                pending.add((ptr,direction))
                                break

                            #print("added")
                            current_outs[other_q] = ptr.next(other_q)
                            buffer.append((other_q, ptr.next(other_q), True))
                            current_ins[other_q] = ptr.prev(other_q)
                            buffer.append((other_q, ptr.prev(other_q), False))

                        elif ptr in [i[0] for i in pending]:
                            _, dir = [p for p in pending if p[0] == ptr][0]
                            pending.remove((ptr,dir))
                            #print("pended",other_q,ptr,dir)
                            if dir:
                                current_outs[other_q] = ptr.next(other_q)
                                buffer.append((other_q, current_outs[other_q], dir))
                            else:
                                current_ins[other_q] = ptr.prev(other_q)
                                buffer.append((other_q, current_ins[other_q], dir))
                        else:
                            pending.add((ptr,direction))
                            #print("pending",ptr,direction)
                            current_bound[qubit] = ptr
                            break
                    elif ptr.nodeType == "rz":
                        if maxsize != None and cost(len(current_ins.keys()), rotation_count+1) > maxsize:
                            break
                        rotation_count += 1
                    tracked.add(ptr)
                    if direction:
                        ptr = ptr.next(qubit)
                    else:
                        ptr = ptr.prev(qubit)
                    current_bound[qubit] = ptr
                    '''print(buffer)
                    print("cin",current_ins)
                    print("cout",current_outs)
                    print("pending",pending)
                    print("@", Circuit(ins=current_ins, outs=current_outs))'''
                if ptr == None:
                    raise Exception("Out of bounds (this shouldn't happen)")

            while len(buffer) > 0:
                '''print()
                print("keys:", current_ins.keys())
                print("buffer:",buffer)
                print("pending:",pending)
                print("current_ins:",current_ins)
                print("current_outs:",current_outs)
                
                sub_seq = Circuit(ins=current_ins, outs=current_outs).get_sequence()
                for node in this.get_sequence():
                    if node in sub_seq:
                        print(">"+str(node))
                    else:
                        print(node)'''
                
                q,n,d = buffer.popleft()
                parse_anchor(q,n,d)
            
            '''
            print("\nFinal:")
            print("buffer:",buffer)
            print("pending:",pending)
            print("current_ins:",current_ins)
            print("current_outs:",current_outs)
            print(Circuit(ins=current_ins, outs=current_outs))
            sub_seq = Circuit(ins=current_ins, outs=current_outs).get_sequence()
            for node in this.get_sequence():
                if node in sub_seq:
                    print(">"+str(node))
                else:
                    print(node)
            '''
            stats.append(Circuit(ins=current_ins, outs=current_outs).get_stats())
            Circuit(ins=current_ins, outs=current_outs).replace_with_phasePoly(preserve_original=preserve_original)
            '''print(Circuit(ins=current_ins, outs=current_outs))
            this.get_sequence()
            print("~~~~~~~~~")'''
        return stats

    #TODO  code is not optimized for combined circuits
    def partition_to_phasePoly_wide(this, preserve_original=False, size_metric:Literal['qibit_rotation_product','qubits','rotations']="qibit_rotation_product", maxsize:int|None = None):
        """
        Traverse the circuit to group consecutive legal gates (RZ, CNOT/CX, SWAP)
        into phase-polynomial blocks.

        Algorithm sketch:
        - Start from each unvisited legal node and expand a block along all involved
        qubits in both forward/backward directions.
        - While expanding:
            * Include single-qubit RZs (counted for size checks).
            * For two-qubit gates, if the partner wire is not yet in the block,
            run quick forward/backward scans to avoid dependency cycles:
                - If safe, add that wire and continue expansion.
                - Otherwise mark the gate as pending and revisit once the other side
                is reached.
        - Continue until no more anchors are left; then replace this subcircuit by a
        phase-polynomial form and collect stats.

        Traversal uses a work queue (for anchors) and a pending set (for deferred gates),
        so each gate may be revisited when its partner wire is later pulled in.

        Complexity:
        - Each legal node is processed once, with possible small forward/backward scans.
        - Typical runtime ~O(G), worst-case O(G^2) for G legal gates.
        
        TODO: code can be implemented as a simplified floating window search, it should achive ~O(G) complexity.
        Assume the circuits as a dependency graph, all RZ can be expanded as soon as it appears, all H can be used to build a block barrier as soon as it appears.
        In a dependency graph, the legal two-qubit gates can be expanded as soon as it appears, and the inlegal two-qubit gates can be found after check the target/control wire whether is close or not.
        In wrost case, we just need to visit every node one-two times, it should be O(G).
        """
        stats = []
        #TODO
        tracked = set()

        curtain = this.ins.copy()
        old_curtain = curtain.copy()

        for q in this.qubits:
            curtain[q] = curtain[q].next(q)
            if curtain[q].nodeType == "rz" and curtain[q].next(q).nodeType == "h":
                tracked.add(curtain[q])
                curtain[q] = curtain[q].next(q)
        if len(tracked) > 0:
            print(curtain)
            Circuit(ins=old_curtain.copy(), outs=curtain.copy()).replace_with_phasePoly(preserve_original=preserve_original)
        
        #TODO 
        while True:
            print(curtain)
            end_flag = True
            for q in this.qubits:
                while curtain[q].nodeType not in phasepoly_legal_gates and curtain[q] != this.outs[q]:
                    #print("h shift")
                    tracked.add(curtain[q])
                    curtain[q] = curtain[q].next(q)
                if curtain[q] != this.outs[q]:
                    end_flag = False

            if end_flag:
                break
            
            old_curtain = dict([(k, v.prev(k)) for k,v in curtain.items()])
            seq_nums = dict((b,a) for a,b in enumerate(this.get_sequence()))
            rotation_count = 0

            pending:set[tuple[Node,bool]] = set()
            buffer = deque()
            for q in this.qubits:
                buffer.append((q, curtain[q], True))

            def cost(qubits, rotations):
                if size_metric == 'qubits':
                    return qubits
                elif size_metric == 'rotations':
                    return rotations
                else:
                    return qubits*rotations

            def parse_anchor(qubit:int, node:Node, direction:bool):
                nonlocal rotation_count
                ptr = node
                if direction:
                    current_bound = curtain
                else:
                    current_bound = old_curtain
                while ptr != None and ptr.nodeType in phasepoly_legal_gates:
                    if ptr.nodeType in ("cnot", "cx", "swap"):
                        other_q = ptr.qubits[0] if ptr.qubits[0] != qubit else ptr.qubits[1] 
                        if other_q not in curtain.keys():
                            '''if maxsize != None and cost(len(current_ins.keys())+1, rotation_count) > maxsize:
                                break'''
                            min_out_num = min(seq_nums.get(n,math.inf) for n in curtain.values())
                            max_in_num = max(seq_nums.get(n,-1) for n in old_curtain.values())

                            #print("min_out_num", min_out_num, "max_in_num", max_in_num, ptr, other_q)

                            #anti self-dependency measures

                            stop_flag = False
                            backbuffer = deque([ptr.prev(other_q)])
                            tracked_b = set()
                            while len(backbuffer) > 0:
                                #print("backbuffer", [f"({q},{seq_nums.get(q)})" for q in backbuffer])
                                item = backbuffer.popleft()
                                tracked_b.add(item)
                                if item == None:
                                    raise Exception("Out of bounds (this shouldn't happen)")
                                if item in curtain.values():
                                    #print(f"back stopped at {item}")
                                    stop_flag = True
                                    break
                                if item in seq_nums.keys() and seq_nums[item] > min_out_num:
                                    for o in item.ins.values():
                                        if o not in tracked_b and o not in backbuffer:
                                            backbuffer.append(o)
                            if stop_flag:
                                pending.add((ptr,direction))
                                break
                            forwardbuffer = deque([ptr.next(other_q)])
                            tracked_f = set()
                            while len(forwardbuffer) > 0:
                                #print("forwardbuffer", [f"({q},{seq_nums.get(q)})" for q in forwardbuffer])
                                item = forwardbuffer.popleft()
                                tracked_f.add(item)
                                if item == None:
                                    raise Exception("Out of bounds (this shouldn't happen)")
                                if item in old_curtain.values():
                                    #print(f"forward stopped at {item}")
                                    stop_flag = True
                                    break
                                if item in seq_nums.keys() and seq_nums[item] < max_in_num:
                                    for o in item.outs.values():
                                        if o not in tracked_f and (o not in forwardbuffer):
                                            forwardbuffer.append(o)
                            if stop_flag:
                                pending.add((ptr,direction))
                                break

                            #print("added")
                            curtain[other_q] = ptr.next(other_q)
                            buffer.append((other_q, ptr.next(other_q), True))
                            old_curtain[other_q] = ptr.prev(other_q)
                            buffer.append((other_q, ptr.prev(other_q), False))

                        elif ptr in [i[0] for i in pending]:
                            _, dir = [p for p in pending if p[0] == ptr][0]
                            pending.remove((ptr,dir))
                            #print("pended",other_q,ptr,dir)
                            if dir:
                                curtain[other_q] = ptr.next(other_q)
                                buffer.append((other_q, curtain[other_q], dir))
                            else:
                                old_curtain[other_q] = ptr.prev(other_q)
                                buffer.append((other_q, old_curtain[other_q], dir))
                        else:
                            pending.add((ptr,direction))
                            #print("pending",ptr,direction)
                            current_bound[qubit] = ptr
                            break
                    elif ptr.nodeType == "rz":
                        if maxsize != None and cost(len(curtain.keys()), rotation_count+1) > maxsize:
                            break
                        rotation_count += 1
                    tracked.add(ptr)
                    if direction:
                        ptr = ptr.next(qubit)
                    else:
                        ptr = ptr.prev(qubit)
                    current_bound[qubit] = ptr
                    '''print(buffer)
                    print("cin",current_ins)
                    print("cout",current_outs)
                    print("pending",pending)
                    print("@", Circuit(ins=current_ins, outs=current_outs))'''
                if ptr == None:
                    raise Exception("Out of bounds (this shouldn't happen)")

            while len(buffer) > 0:
                '''print()
                print("keys:", current_ins.keys())
                print("buffer:",buffer)
                print("pending:",pending)
                print("current_ins:",current_ins)
                print("current_outs:",current_outs)
                
                sub_seq = Circuit(ins=current_ins, outs=current_outs).get_sequence()
                for node in this.get_sequence():
                    if node in sub_seq:
                        print(">"+str(node))
                    else:
                        print(node)'''
                
                q,n,d = buffer.popleft()
                parse_anchor(q,n,d)
            
            '''
            print("\nFinal:")
            print("buffer:",buffer)
            print("pending:",pending)
            print("current_ins:",current_ins)
            print("current_outs:",current_outs)
            print(Circuit(ins=current_ins, outs=current_outs))
            sub_seq = Circuit(ins=current_ins, outs=current_outs).get_sequence()
            for node in this.get_sequence():
                if node in sub_seq:
                    print(">"+str(node))
                else:
                    print(node)
            '''
            stats.append(Circuit(ins=old_curtain.copy(), outs=curtain.copy()).get_stats())
            Circuit(ins=old_curtain.copy(), outs=curtain.copy()).replace_with_phasePoly(preserve_original=preserve_original)
            '''print(Circuit(ins=current_ins, outs=current_outs))
            this.get_sequence()
            print("~~~~~~~~~")'''
    
        return stats

    #Note: I tried to run the below on quartz/circuit/nam_rm_circs/gf2^32_mult.qasm
    #It did not terminate despite runnning for more than a few days.

    #Call only for circuit consisting only of Rz, CNOT/CX, and NOT/X gates
    #Untested, use at your own risk
    def convert_to_phasePoly(this, preserve_original=False):
        subcircuit_copy = Circuit(qubits=this.qubits)

        hanging_nots = defaultdict(lambda:False)
        rotations:list[tuple[float|utils.PiAngle, set[int]]] = []
        affineOut:dict[int, set[int]] = dict()
        for q in this.qubits:
            affineOut[q] = {q}

        out = Circuit(qubits = this.qubits)
        for node in this.get_sequence():
            if not isinstance(node, Gate):
                continue
            if node.nodeType == "cx" or node.nodeType == "cnot":
                q0,q1 = node.qubits[0:2]
                affineOut[q1] = affineOut[q0] ^ affineOut[q1]
                hanging_nots[q1] = hanging_nots[q0] ^ hanging_nots[q1]
                subcircuit_copy.append_node(Gate('cx', [q0,q1]))
            elif node.nodeType == "swap":
                q0,q1 = node.qubits[0:2]
                temp = affineOut[q0]
                affineOut[q0] = affineOut[q1]
                affineOut[q1] = temp
                temp = hanging_nots[q0]
                hanging_nots[q0] = hanging_nots[q1]
                hanging_nots[q1] = temp
                subcircuit_copy.append_node(Gate('swap', [q0,q1]))
            elif node.nodeType == "x" or node.nodeType == "not":
                hanging_nots[node.qubits[0]] = not hanging_nots[node.qubits[0]]
            elif node.nodeType == "rz":
                q = node.qubits[0]
                deg = node.params[0]
                if hanging_nots[q]:
                    deg = -deg
                subcircuit_copy.append_node(Gate('rz', [q],[deg]))
                parity = affineOut[q].copy()
                for i in range(len(rotations)):
                    if rotations[i][1] == parity:
                        rotations[i] = (rotations[i][0]+deg, parity)
                        break
                else:
                    rotations.append((deg, parity))
            elif node.nodeType == "i":
                pass
            else:
                raise Exception(f"Illegal gate in subcircuit for phase polynomial: {node}\n{this}")

        rotations = [t for t in rotations if t[0] != 0.0]

        poly_qubits = this.qubits.copy()
        
        #This code seems to create some petty issues that are hard to resolve, namely circuit reversion. Consider it for the future.
        shrinked = False
        if not preserve_original:
            for q in this.qubits: 
                if any((q in r) for _,r in rotations):
                    continue
                if any((q != k and q in v) for k,v in affineOut.items()):
                    continue
                if affineOut[q] != {q}:
                    continue
                poly_qubits.remove(q)
                affineOut.pop(q)
                shrinked = True
        
        phasepoly_gate = PhasePoly(poly_qubits,rotations,affineOut)
        if not shrinked:
            phasepoly_gate.possible_circuits.append(subcircuit_copy)
        out.append_node(phasepoly_gate)

        for k,v in hanging_nots.items():
            if v:
                out.append_node(Gate("x",[k]))
        return out

    def replace_with_phasePoly(this, preserve_original=False):
        pp:Circuit = this.convert_to_phasePoly(preserve_original=preserve_original)
        '''print("+=+")
        print(this)
        print("PP: ",pp)'''
        pp.replace_at(this)

    def commentbound_phasepolys(this):
        prev_nodes = this.ins.copy()
        phasepoly_count = 1
        for node in this.get_sequence():
            if node.nodeType == 'phasePoly':
                ffront = CommentLine(this.qubits.copy(), "")
                ffront.attach_after(prev_nodes)
                for q in ffront.qubits:
                    prev_nodes[q] = ffront
                front = CommentLine(this.qubits.copy(), f"vvvvvvvvvv Partition {phasepoly_count} vvvvvvvvvv {node.__repr__()}")
                front.attach_after(prev_nodes)
                for q in front.qubits:
                    prev_nodes[q] = front
            for q in node.qubits:
                prev_nodes[q] = node
            if node.nodeType == 'phasePoly':
                back = CommentLine(this.qubits.copy(), f"^^^^^^^^^^ Partition {phasepoly_count} ^^^^^^^^^^")
                back.attach_after(prev_nodes)
                phasepoly_count += 1
                for q in back.qubits:
                    prev_nodes[q] = back
                bback = CommentLine(this.qubits.copy(), "")
                bback.attach_after(prev_nodes)
                for q in bback.qubits:
                    prev_nodes[q] = bback

    def remove_commentbounds(this):
        for node in this.get_sequence():
            if node.nodeType == "commentLine":
                node.remove_from_circuit()

    def s_and_t_to_rz(this):
        for g in this.get_sequence():
            if not isinstance(g, Gate):
                continue
            if g.nodeType == "t":
                g.nodeType = "rz"
                g.params = [utils.PiAngle(0.25)]
            elif g.nodeType == "tdg":
                g.nodeType = "rz"
                g.params = [utils.PiAngle(-0.25)]
            elif g.nodeType == "s":
                g.nodeType = "rz"
                g.params = [utils.PiAngle(0.5)]
            elif g.nodeType == "sdg":
                g.nodeType = "rz"
                g.params = [utils.PiAngle(-0.5)]
    
    def convert_rz_to_phase_gates(this, tolerance: float = 1e-9):
        """
        Convert qualifying RZ gates into Z/S/T family gates based on their angles.

        Rules:
            - π multiples become `z`
            - π/2 multiples become `s` or `sdg`
            - π/4 odd multiples become `t` or `tdg`
            - Zero-angle rotations are removed
            - Any angle that is not a multiple of π, π/2, or π/4 are `rz_other`

        Args:
            tolerance: Numerical tolerance when comparing angle multiples.

        Returns:
            A statistics dict with how many gates were converted/removed/left as rz_other.

        Raises:
            ValueError: If an RZ gate parameter cannot be interpreted as a multiple of π.
        """

        def _normalize_multiplier(angle_value) -> float:
            if angle_value is None:
                raise ValueError("Encountered RZ gate with missing angle parameter.")
            if isinstance(angle_value, utils.PiAngle):
                mult = angle_value.get_normalized_mult()
            elif isinstance(angle_value, (int, float)):
                mult = (angle_value / math.pi)
            elif isinstance(angle_value, str):
                cleaned = angle_value.strip().lower()
                if cleaned.endswith("*pi"):
                    coeff_str = cleaned[:-3].strip()
                    if coeff_str in ("", "+"):
                        mult = 1.0
                    elif coeff_str == "-":
                        mult = -1.0
                    else:
                        try:
                            mult = float(coeff_str)
                        except ValueError as exc:
                            raise ValueError(f"Unsupported RZ angle format: '{angle_value}'") from exc
                elif cleaned in ("pi", "+pi"):
                    mult = 1.0
                elif cleaned == "-pi":
                    mult = -1.0
                else:
                    raise ValueError(f"Unsupported RZ angle format: '{angle_value}'")
            else:
                raise ValueError(f"Unsupported RZ angle type: {type(angle_value).__name__}")

            while mult > 1:
                mult -= 2
            while mult <= -1:
                mult += 2
            return mult

        def _is_close(value: float, target: float) -> bool:
            return abs(value - target) < tolerance

        stats = {
            "removed_zero": 0,
            "z": 0,
            "s": 0,
            "sdg": 0,
            "t": 0,
            "tdg": 0,
            "rz_other": 0,
        }

        for gate in list(this.get_sequence()):
            if not isinstance(gate, Gate):
                continue
            if gate.nodeType != "rz" or not gate.params:
                continue
            multiplier = _normalize_multiplier(gate.params[0])
            if _is_close(multiplier, 0.0):
                gate.remove_from_circuit()
                stats["removed_zero"] += 1
                continue
            if _is_close(abs(multiplier), 1.0):
                gate.nodeType = "z"
                gate.params = []
                stats["z"] += 1
                continue
            if _is_close(abs(multiplier), 0.5):
                gate.nodeType = "s" if multiplier > 0 else "sdg"
                gate.params = []
                stats["s" if multiplier > 0 else "sdg"] += 1
                continue
            if (_is_close(abs(multiplier), 0.25) or
                _is_close(abs(multiplier), 0.75)):
                gate.nodeType = "t" if multiplier > 0 else "tdg"
                gate.params = []
                stats["t" if multiplier > 0 else "tdg"] += 1
                continue
            gate.nodeType = "rz_other"
            stats["rz_other"] += 1

        return stats
    
    def condense_cnots_to_swaps(this):
        for gate in this.get_sequence():
            if gate.nodeType != 'cx':
                continue
            q0, q1 = gate.qubits
            prvgate = gate.prev(q0)
            if prvgate != gate.prev(q1):
                continue
            if prvgate.nodeType not in ('cx','cnot'):
                continue
            if list(prvgate.qubits) != [q1, q0]:
                continue
            prvprvgate = prvgate.prev(q0)
            if prvprvgate != prvgate.prev(q1):
                continue
            if prvprvgate.nodeType not in ('cx','cnot'):
                continue
            if list(prvprvgate.qubits) != [q0, q1]:
                continue
            prvprvgate.remove_from_circuit()
            prvgate.remove_from_circuit()
            gate.nodeType = "swap"

    def cancel_cx_gates(this):
        for gate in this.get_sequence():
            if gate.nodeType != 'cx':
                continue

            q0, q1 = gate.qubits
            
            mark = None
            prv0 = gate.prev(q0)
            while True:
                if prv0.nodeType != 'cx' or prv0.qubits[0] != q0:
                    break
                if prv0.qubits[1] == q1:
                    mark = prv0
                    break
                prv0 = prv0.prev(q0)
                continue
                
            if mark == None:
                continue
            prv1 = gate.prev(q1)
            while True:
                if prv1.nodeType != 'cx' or prv1.qubits[1] != q1:
                    break
                if prv1.qubits[0] == q0: 
                    if prv1 == mark:
                        gate.remove_from_circuit()
                        mark.remove_from_circuit()
                    break
                prv1 = prv1.prev(q1)
                continue
        #print(this)

    def cancel_swap_gates(this):
        for gate in this.get_sequence():
            if gate.nodeType != 'swap':
                continue
            q0, q1 = gate.qubits
            prvgate = gate.prev(q0)
            if prvgate != gate.prev(q1):
                continue
            if prvgate.nodeType != 'swap':
                continue
            gate.remove_from_circuit()
            prvgate.remove_from_circuit()
        #print(this)

    #obsolete
    def pull_rz_gates(this):
        for gate in this.get_sequence():
            if gate.nodeType != "rz":
                continue
            gate:Gate
            gate_q = gate.qubits[0]
            while True:
                prev = gate.prev(gate_q)
                if prev.nodeType == "cx" or prev.nodeType == "cnot":
                    if gate_q != prev.qubits[0]:
                        break
                    #print(f"{gate} <--> {prev}")
                    gate.shift_backward(gate_q)
                elif prev.nodeType == "rz":
                    #print(f"{gate} <--> {prev}")
                    gate.remove_from_circuit()
                    prev:Gate
                    prev.params[0] += gate.params[0]
                    if prev.params[0] == 0:
                        prev.remove_from_circuit()
                    break
                else:
                    prev2 = prev.prev(gate_q)
                    if prev2 == None:
                        break
                    prev3 = prev2.prev(gate_q)
                    if prev3 == None:
                        break
                    if prev.nodeType == "h" and prev3.nodeType == "h" and prev2.nodeType == "cx":
                        if prev2.qubits[1] != gate_q:
                            break
                        #print(f"{gate} <--> {prev}")
                        prev.connect_to(gate.next(gate_q), gate_q)
                        gate.connect_from(prev3.prev(gate_q), gate_q)
                        prev3.connect_from(gate, gate_q)
                    elif prev.nodeType == "cx" and prev3.nodeType == "cx" and prev2.nodeType == "rz":
                        if prev.qubits[1] != gate_q or prev3.qubits[1] != gate_q:
                            break
                        if prev.prev(gate_q) != prev3:
                            break
                        #print(f"{gate} <--> {prev}")
                        prev.connect_to(gate.next(gate_q), gate_q)
                        gate.connect_from(prev3.prev(gate_q), gate_q)
                        prev3.connect_from(gate, gate_q)
                    else:
                        break

    #obsolete
    def push_x_gates(this):
        def _push_single_x_gate(xgate:Gate):
            #print("pushing:", xgate)
            q = xgate.qubits[0]
            while True:
                next = xgate.next(q)
                #print("loop", this, '\n')
                if next.nodeType in ("cx", "cnot"):
                    #print(f"{xgate} <--> {next}")
                    if next.qubits[0] == q:
                        new_xgate = Gate("x", [next.qubits[1]])
                        new_xgate.attach_after(dict([(next.qubits[1], next)]))
                        #print("xgate insert:", new_xgate, this, '\n')
                        _push_single_x_gate(new_xgate)
                    xgate.shift_forward(q)
                elif next.nodeType == "swap":
                    #print(f"{xgate} <--> {next}")
                    other_q = next.qubits[0] if next.qubits[0] != q else next.qubits[1]
                    xgate.remove_from_circuit()
                    xgate.qubits[0] = other_q
                    q = other_q
                    xgate.attach_after(dict([(q, next)]))
                elif next.nodeType == "rz":
                    #print(f"{xgate} <--> {next}")
                    if not isinstance(next, Gate):
                        continue
                    next.params[0] *= -1
                    xgate.shift_forward(q)
                elif next.nodeType in ("x", "not"):
                    #print(f"{xgate} <--> {next}")
                    xgate.remove_from_circuit()
                    next.remove_from_circuit()
                    break
                elif next.nodeType == "h":
                    #print(f"{xgate} <--> {next}")
                    if not isinstance(next, Gate):
                        continue
                    xgate.shift_forward(q)
                    xgate.nodeType = "rz"
                    xgate.params = [utils.PiAngle(1)]
                    break
                else:
                    break

        #print(this, "\n")
        for q in this.qubits:
            ptr = this.ins[q]
            while ptr.next(q) not in (this.outs[q], None):
                nxtptr = ptr.next(q)
                #print('nxtptr', nxtptr)
                if nxtptr.nodeType == "x":
                    if not isinstance(nxtptr, Gate):
                        continue
                    _push_single_x_gate(nxtptr)
                    #print(this)
                    #print("-")
                ptr = ptr.next(q)
        #print(this)

    def pull_x_gates(this, condense_rzs=True):
        def _pull_rz(rzgate:Gate):
            if rzgate.nodeType != "rz":
                raise Exception("Incorrect input parameter")
            gate_q = rzgate.qubits[0]
            while True:
                prev = rzgate.prev(gate_q)
                if prev.nodeType == "cx" or prev.nodeType == "cnot":
                    if gate_q != prev.qubits[0]:
                        break
                    #print(f"{gate} <--> {prev}")
                    rzgate.shift_backward(gate_q)
                elif prev.nodeType == "rz":
                    #print(f"{gate} <--> {prev}")
                    rzgate.remove_from_circuit()
                    if not isinstance(prev, Gate):
                        continue
                    prev.params[0] += rzgate.params[0]
                    if prev.params[0] == 0:
                        prev.remove_from_circuit()
                    break
                else:
                    break
        def _pull_single_x_gate(xgate:Gate):
            #print("pushing:", xgate)
            q = xgate.qubits[0]
            while True:
                prev = xgate.prev(q)
                #print("loop", this, '\n')
                if prev.nodeType in ("cx", "cnot"):
                    #print(f"{xgate} <--> {prev}")
                    if prev.qubits[0] == q:
                        new_xgate = Gate("x", [prev.qubits[1]])
                        new_xgate.attach_before(dict([(prev.qubits[1], prev)]))
                        #print("xgate insert:", new_xgate, this, '\n')
                        _pull_single_x_gate(new_xgate)
                    xgate.shift_backward(q)
                elif prev.nodeType == "swap":
                    #print(f"{xgate} <--> {prev}")
                    other_q = prev.qubits[0] if prev.qubits[0] != q else prev.qubits[1]
                    xgate.remove_from_circuit()
                    xgate.qubits[0] = other_q
                    q = other_q
                    xgate.attach_before(dict([(q, prev)]))
                elif prev.nodeType == "rz":
                    #print(f"{xgate} <--> {prev}")
                    if not isinstance(prev, Gate):
                        continue
                    prev.params[0] *= -1
                    xgate.shift_backward(q)
                elif prev.nodeType in ("x", "not"):
                    #print(f"{xgate} <--> {prev}")
                    xgate.remove_from_circuit()
                    prev.remove_from_circuit()
                    break
                elif prev.nodeType == "h":
                    #print(f"{xgate} <--> {prev}")
                    xgate.shift_backward(q)
                    xgate.nodeType = "rz"
                    xgate.params = [utils.PiAngle(1)]
                    if condense_rzs:
                        _pull_rz(xgate)
                    break
                else:
                    break

        #print(this, "\n")
        for q in this.qubits:
            ptr = this.outs[q]
            # Walk backwards from the output towards the input on each qubit line.
            while True:
                prvptr = ptr.prev(q)
                # Stop if we reached the input sentinel or there is no previous node.
                if prvptr in (this.ins[q], None):
                    break
                #print('prvptr', prvptr)
                if prvptr.nodeType == "x":
                    if not isinstance(prvptr, Gate):
                        continue
                    nxt = prvptr.next(q)
                    # 1) If the next gate is also an X/NOT on this qubit, cancel both and stay at current ptr.
                    if nxt is not None and nxt not in (this.outs[q],) and nxt.nodeType in ("x", "not"):
                        prvptr.remove_from_circuit()
                        nxt.remove_from_circuit()
                    else:
                        prev_node = prvptr.prev(q)
                        # 2) If the previous gate is an H gate, always pull this X (convert to RZ(pi)),
                        #    regardless of whether it is currently the last gate.
                        if prev_node is not None and prev_node not in (this.ins[q],) and prev_node.nodeType == "h":
                            _pull_single_x_gate(prvptr)
                        else:
                            # 3) Otherwise, only pull X when it is not the last gate on this qubit line
                            #    (i.e., there is a real gate after it, not just the output sentinel).
                            if nxt not in (this.outs[q], None):
                                _pull_single_x_gate(prvptr)
                                #print(this)
                                #print("-")
                # Move one step backward on this qubit line, guarding against None.
                ptr_prev = ptr.prev(q)
                if ptr_prev in (this.ins[q], None):
                    break
                ptr = ptr_prev
        #print(this)
        
    def pull_h_gates(this):
        def _pull_single_h_gate(hgate:Gate):
            #print("pushing:", hgate)
            q = hgate.qubits[0]
            while True:
                prev = hgate.prev(q)
                #print("loop", hgate, prev, '\n')
                if prev.nodeType in ("h"):
                    #print(f"{hgate} <--> {prev}")
                    hgate.remove_from_circuit()
                    prev.remove_from_circuit()
                    break
                elif prev.nodeType == "swap":
                    #print(f"{hgate} <--> {prev}")
                    other_q = prev.qubits[0] if prev.qubits[0] != q else prev.qubits[1]
                    hgate.remove_from_circuit()
                    hgate.qubits[0] = other_q
                    q = other_q
                    hgate.attach_before(dict([(q, prev)]))
                else:
                    break

        #print(this, "\n")
        for q in this.qubits:
            ptr = this.outs[q]
            while ptr.prev(q) not in (this.ins[q], None):
                prvptr = ptr.prev(q)
                #print('prvptr', prvptr)
                if prvptr.nodeType == "h":
                    _pull_single_h_gate(prvptr)
                    #print(this)
                    #print("-")
                ptr = ptr.prev(q)
        #print(this)

    # Transform Hadamard-CX-Hadamard patterns:
    # 1. h[x] cx[y,x] h[x] -> h[y] cx[x,y] h[y]
    # 2. h[x] cx[x,y] h[x] -> h[y] cx[y,x] h[y]
    # 3. h[x] h[y] cx[y,x] h[x] h[y] -> (cancels to empty)
    # Note: This should be used after pull_x_gates since we don't move h gates across x gates
    def transform_cx_h_gates(this):
        def _is_all_h_gates(node:Node, qubit:int):
            """
            Check if a qubit has Hadamard gates both before and after a given node.
            Args:
                node: The current node being examined
                qubit: The qubit to check for surrounding Hadamard gates
            Returns:
                bool: True if Hadamard gates exist both before and after the node and connect to the same nodes
            """
            h_before = node.prev(qubit)
            h_after = node.next(qubit)
            
            # Both surrounding gates must be Hadamard gates
            if h_before.nodeType != "h" or h_after.nodeType != "h":
                return False
                
            return True
        
        for node in this.get_sequence():
            if node.nodeType != "cx":
                continue
            control = node.qubits[0]
            target = node.qubits[1]
            # Case 1: Hadamard gates on both control and target qubits - cancel all 5 gates
            if _is_all_h_gates(node, target) and _is_all_h_gates(node, control):
                node.prev(target).remove_from_circuit()
                node.next(target).remove_from_circuit()
                node.prev(control).remove_from_circuit()
                node.next(control).remove_from_circuit()
                node.remove_from_circuit()
            # Case 2: Hadamard gates only on target qubit - flip control/target and relocate Hadamard
            elif _is_all_h_gates(node, target):
                # Remove original gates
                node.prev(target).remove_from_circuit()
                node.next(target).remove_from_circuit()
                # Create and connect new gates with flipped orientation
                new_h_before = Gate("h", [control])
                new_h_before.attach_before(dict([(control, node)]))
                new_h_after = Gate("h", [control])
                new_h_after.attach_after(dict([(control, node)]))
                #flip original node
                node.qubits = [target, control]
          # Case 3: Hadamard gates only on control qubit - flip control/target and relocate Hadamard
            elif _is_all_h_gates(node, control):
                # Remove original gates
                node.prev(control).remove_from_circuit()
                node.next(control).remove_from_circuit()
                # Create and connect new gates with flipped orientation
                new_h_before = Gate("h", [target])
                new_h_before.attach_before(dict([(target, node)]))
                new_h_after = Gate("h", [target])
                new_h_after.attach_after(dict([(target, node)]))
                #flip original node
                node.qubits = [target, control]

    def pull_xh_gates(this):
        this.pull_x_gates()
        this.pull_h_gates()

    def circuit_wide_rz_floating(this, pure_rotation_merging:bool = False):
        # if not in pure rotation merging mode, pull x gates to cancel more x gates
        if not pure_rotation_merging:
            this.pull_x_gates()
        #print()
        current_parities = dict((q, {q}) for q in this.qubits)
        first_positions:list[tuple[set[int],Node]] = []
        h_pairs:list[tuple[set[int], int]] = []
        next_fake_qubit = -1
        for node in this.get_sequence():
            #detect new positions and stuff
            #print(node)
            #print("current_parities", current_parities)
            
            if node.nodeType != "rz":
                for q, p in current_parities.items():
                    if q not in node.qubits or p in [fp[0] for fp in first_positions]:
                        continue
                    new_rz = Gate("rz", [q], [0.0])
                    new_rz.attach_after(dict([(q, node.prev(q))]))
                    first_positions.append((p, new_rz))
                    #print("f", (p, new_rz))

            if node.nodeType in ("cx","cnot"):
                current_parities[node.qubits[1]] = current_parities[node.qubits[0]].symmetric_difference(current_parities[node.qubits[1]])
            elif node.nodeType == "swap":
                temp = current_parities[node.qubits[1]]
                current_parities[node.qubits[1]] = current_parities[node.qubits[0]]
                current_parities[node.qubits[0]] = temp
            elif node.nodeType == "rz":
                #TODO
                node:Gate
                for p,n in first_positions:
                    if current_parities[node.qubits[0]] == p:
                        n:Gate
                        n.params[0] += node.params[0]
                        node.remove_from_circuit()
                        break
                else:
                    first_positions.append((current_parities[node.qubits[0]], node))
            else:
                for q in node.qubits:
                    '''if node.nodeType == "h":
                        if len(current_parities[q]) == 1:
                            (result_q,) = current_parities[q]
                            print("testing: ", result_q, h_pairs)
                            try:
                                index = [h[1] for h in h_pairs].index(result_q)
                                current_parities[q] = h_pairs[index][0]
                                break
                            except ValueError:
                                pass
                        try:
                            index = [h[0] for h in h_pairs].index(current_parities[q])
                            current_parities[q] = {h_pairs[index][1]}
                            break
                        except ValueError:
                            pass
                        h_pairs.append((current_parities[q], next_fake_qubit))'''
                    current_parities[q] = {next_fake_qubit}
                    next_fake_qubit -= 1
        #print(first_positions)
        for _,n in first_positions:
            if n.params[0] == 0:
                n.remove_from_circuit()
        # if not in pure rotation merging mode, pull h gates to cancel more h gates
        if not pure_rotation_merging:
            this.pull_h_gates()


    def remove_start_swaps_in_place(this):
        unlocked = set(this.qubits)
        trailing_nodes:defaultdict[int,list[Node]] = defaultdict(list)
        for n in this.get_sequence():
            if len(unlocked) == 0:
                break
            if len(n.qubits) == 1:
                if n.qubits[0] not in unlocked:
                    continue
                trailing_nodes[n.qubits[0]].insert(0, n)
                n.remove_from_circuit()
                continue
            if n.nodeType == 'swap': #TODO 
                i,j = n.qubits
                if i not in unlocked or j not in unlocked:
                    continue
                temp = trailing_nodes[i]
                trailing_nodes[i] = trailing_nodes[j]
                trailing_nodes[j] = temp

                n.remove_from_circuit()
                continue
            for q in n.qubits:
                unlocked.discard(q)
        for qubit,nodes in trailing_nodes.items():
            for n in nodes:
                if n.nodeType == "phasePoly":
                    n:PhasePoly
                    n.replace_qubit(n.qubits[0], qubit)
                else:
                    n.qubits = [qubit]
                this.prepend_node(n)
            pass

    def remove_end_swaps_in_place(this):
        unlocked = set(this.qubits)
        trailing_nodes:defaultdict[int,list[Node]] = defaultdict(list)
        for n in reversed(this.get_sequence()):
            if len(unlocked) == 0:
                break
            if len(n.qubits) == 1:
                if n.qubits[0] not in unlocked:
                    continue
                trailing_nodes[n.qubits[0]].insert(0, n)
                n.remove_from_circuit()
                continue
            if n.nodeType == 'swap': #TODO 
                i,j = n.qubits
                if i not in unlocked or j not in unlocked:
                    continue
                temp = trailing_nodes[i]
                trailing_nodes[i] = trailing_nodes[j]
                trailing_nodes[j] = temp

                n.remove_from_circuit()
                continue
            for q in n.qubits:
                unlocked.discard(q)
        for qubit,nodes in trailing_nodes.items():
            for n in nodes:
                if n.nodeType == "phasePoly":
                    n:PhasePoly
                    n.replace_qubit(n.qubits[0], qubit)
                else:
                    n.qubits = [qubit]
                this.append_node(n)
            pass

    def __repr__(this) -> str:
        out = "\n".join([n.__repr__() for n in this.get_sequence()])
        return "\n>"+out
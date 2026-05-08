from typing_extensions import Self, Literal #For annotations only
import heapq
import src.utils as utils
from collections import deque, defaultdict
import numpy as np
import depq
import functools
import copy

from src.circuits import Circuit, Gate, PhasePoly, CommentLine, Node

# Module-level switch selecting which Gaussian-elimination implementation the
# unparameterised ``gaussian_elim`` callers (inside this file) should use.
# Default is the "modified" lookahead-pivot algorithm. ``phasepoly_synthesize``
# may swap this temporarily via ``set_gaussian_elim_algorithm`` so that an
# entire synthesis run uses the historical "classic" algorithm instead.
_GAUSSIAN_ELIM_ALGORITHM: Literal["modified", "classic"] = "modified"


def set_gaussian_elim_algorithm(name: Literal["modified", "classic"]) -> str:
    """Set the module-level Gaussian-elimination algorithm.

    Returns the previous value so the caller can restore it (typically inside a
    try/finally guard).
    """
    global _GAUSSIAN_ELIM_ALGORITHM
    if name not in ("modified", "classic"):
        raise ValueError(
            f"Invalid gaussian_elim algorithm '{name}'; must be 'modified' or 'classic'."
        )
    prev = _GAUSSIAN_ELIM_ALGORITHM
    _GAUSSIAN_ELIM_ALGORITHM = name
    return prev


def get_gaussian_elim_algorithm() -> str:
    return _GAUSSIAN_ELIM_ALGORITHM


def gaussian_elim(
    matrix: np.ndarray,
    row_span: list[int] | None = None,
    col_span: list[int] | None = None,
    *,
    algorithm: Literal["modified", "classic"] | None = None,
):
    """Dispatcher: run either the modified or classic GF(2) Gaussian
    elimination depending on ``algorithm`` (or the module-level default when
    ``algorithm`` is None).

    Both implementations return a list of (control, target) CNOT pairs that
    realise the elimination.
    """
    chosen = algorithm if algorithm is not None else _GAUSSIAN_ELIM_ALGORITHM
    if chosen == "classic":
        return _gaussian_elim_classic(matrix, row_span, col_span)
    if chosen == "modified":
        return _gaussian_elim_modified(matrix, row_span, col_span)
    raise ValueError(
        f"Invalid gaussian_elim algorithm '{chosen}'; must be 'modified' or 'classic'."
    )


def _gaussian_elim_modified(
    matrix: np.ndarray,
    row_span: list[int] | None = None,
    col_span: list[int] | None = None,
):
    """
    GF(2) Gaussian elimination with one-column lookahead pivot selection and
    delta-based diagonal repair. Records each row XOR as a CNOT (control, target).

    Pivot selection:
        For each candidate column, replay one full elimination step on a
        snapshot of the matrix and score the resulting (distance-from-identity
        + ops); the candidate with the lowest score wins. Ties are broken by
        the candidate's distance-from-identity, then by column index.

    Diagonal repair:
        When the pivot diagonal is zero, choose the swap source row that
        minimises the row-XOR delta on the remaining unchecked columns
        (rather than picking the first row found with a 1 in this column).

    Returns:
        list of (control, target) CNOT pairs that realise the elimination.

    Complexity:
        O(n^4) worst case: the per-column lookahead replays a full elimination
        pass for each candidate column. The elimination itself is O(n^3); space
        is O(n^2) for the matrix and up to O(n^2) for the CNOT list.
    """
    if row_span is None:
        row_span = list(range(matrix.shape[0]))
    if col_span is None:
        col_span = row_span

    id_matrix = np.identity(matrix.shape[0], dtype=int)
    cnot_list: list[tuple[int, int]] = []
    checked_cols = set(r for r in row_span if r not in col_span)
    unchecked_cols = set(c for c in col_span if c in row_span)

    while len(unchecked_cols) > 0:
        def row_op_delta(i_src, j_dst, cols):
            old = matrix[j_dst, cols]
            new = old ^ matrix[i_src, cols]
            return (
                int(np.sum(new ^ id_matrix[j_dst, cols]))
                - int(np.sum(old ^ id_matrix[j_dst, cols]))
            )

        if len(unchecked_cols) == 1:
            col = next(iter(unchecked_cols))
        else:
            def _replay_one_col(cand: int) -> tuple[int, int]:
                snap = matrix.copy()
                unchecked_snapshot = set(unchecked_cols)
                checked_snapshot = set(checked_cols)
                unchecked_snapshot.remove(cand)
                ops = 0

                def _rowsum(row: int):
                    return np.sum(snap[row] ^ snap[cand])

                def _row_op_delta_snap(i_src: int, j_dst: int, cols: list[int]) -> int:
                    old = snap[j_dst, cols]
                    new = old ^ snap[i_src, cols]
                    return (
                        int(np.sum(new ^ id_matrix[j_dst, cols]))
                        - int(np.sum(old ^ id_matrix[j_dst, cols]))
                    )

                diag_ok = bool(snap[cand, cand] == 1)

                if not diag_ok:
                    active_now = sorted(unchecked_snapshot)
                    traditional_sorted_rows = sorted(
                        unchecked_snapshot,
                        key=lambda r: (_rowsum(r), r),
                    )
                    traditional_order = {
                        row: rank
                        for rank, row in enumerate(traditional_sorted_rows)
                    }
                    best_src = None
                    best_key = None
                    for src in unchecked_snapshot:
                        if snap[src, cand] != 1:
                            continue
                        d = _row_op_delta_snap(src, cand, active_now)
                        key = (d, traditional_order[src])
                        if best_key is None or key < best_key:
                            best_key, best_src = key, src
                    if best_src is not None:
                        snap[cand, :] ^= snap[best_src, :]
                        ops += 1
                        diag_ok = True

                for row in sorted(unchecked_snapshot, key=_rowsum):
                    if snap[row, cand] == 0:
                        continue
                    snap[row, :] ^= snap[cand, :]
                    ops += 1

                for row in sorted(checked_snapshot):
                    if snap[row, cand] == 0:
                        continue
                    if not diag_ok and row in col_span:
                        raise Exception(f"Non-singular matrix: {snap}")
                    snap[row, :] ^= snap[cand, :]
                    ops += 1

                return int(np.sum(snap ^ id_matrix)), ops

            col_id_dist = np.sum(matrix ^ id_matrix, axis=0)
            best_key = None
            col = -1
            for cand in sorted(unchecked_cols):
                h, ops = _replay_one_col(cand)
                metric = (h + ops,)
                key = (metric, int(col_id_dist[cand]), cand)
                if best_key is None or key < best_key:
                    best_key = key
                    col = cand

        unchecked_cols.remove(col)

        diag_one = False if matrix[col, col] == 0 else True

        def rowsum(row):
            return np.sum(matrix[row] ^ matrix[col])

        if not diag_one:
            active_now = sorted(unchecked_cols)
            traditional_sorted_rows = sorted(
                unchecked_cols,
                key=lambda r: (rowsum(r), r),
            )
            traditional_order = {
                row: rank
                for rank, row in enumerate(traditional_sorted_rows)
            }
            best_src = None
            best_key = None
            for src in unchecked_cols:
                if matrix[src, col] != 1:
                    continue
                d = row_op_delta(src, col, active_now)
                key = (d, traditional_order[src])
                if best_key is None or key < best_key:
                    best_key, best_src = key, src
            if best_src is not None:
                matrix[col, :] ^= matrix[best_src, :]
                cnot_list.append((best_src, col))
                diag_one = True

        for row in sorted(unchecked_cols, key=rowsum):
            if matrix[row, col] == 0:
                continue
            matrix[row, :] ^= matrix[col, :]
            cnot_list.append((col, row))
        for row in checked_cols:
            if matrix[row, col] == 0:
                continue
            if not diag_one and row in col_span:
                raise Exception(f"Non-singular matrix: {matrix}")
            matrix[row, :] ^= matrix[col, :]
            cnot_list.append((col, row))
        checked_cols.add(col)

    return cnot_list


def _gaussian_elim_classic(
    matrix: np.ndarray,
    row_span: list[int] | None = None,
    col_span: list[int] | None = None,
):
    """Historical GF(2) Gaussian elimination (verbatim port of ``guassian_elim``
    from commit 896c33195aa5d722e09f276720de2ab5c0c6a550).

    Pivot selection:
        Greedy by total whole-matrix distance-from-identity for each candidate
        column.

    Diagonal repair:
        Iterate the unchecked rows in ``rowsum``-sorted order and use the first
        one with a 1 in the pivot column to fix the diagonal.

    The historical signature also accepted a ``mapping`` parameter that
    routed pivot selection through a coupling-graph weight; that branch is
    dropped here because ``coupling_graph`` no longer exists in this
    repository and none of the callers ever passed it.

    Returns:
        list of (control, target) CNOT pairs that realise the elimination.

    Complexity:
        O(n^4) worst case (column-cost recomputation), O(n^3) for the
        elimination itself.
    """
    if row_span == None:
        row_span = list(range(matrix.shape[0]))
    if col_span == None:
        col_span = row_span

    id_matrix = np.identity(matrix.shape[0], dtype=int)
    cnot_list: list[tuple[int, int]] = []
    checked_cols = set(r for r in row_span if r not in col_span)
    unchecked_cols = set(c for c in col_span if c in row_span)
    while len(unchecked_cols) > 0:
        col_min = min(np.sum((matrix ^ id_matrix)[:, a]) for a in unchecked_cols)
        valid_cols = np.nonzero(np.sum(matrix ^ id_matrix, 0) == col_min)[0].reshape(-1)
        for c in valid_cols:
            if c in unchecked_cols:
                col = c
                break

        unchecked_cols.remove(col)

        diag_one = False if matrix[col, col] == 0 else True

        def rowsum(row):
            return np.sum(matrix[row] ^ matrix[col])

        for row in sorted(unchecked_cols, key=rowsum):
            if matrix[row, col] == 0:
                continue
            if diag_one == False:
                matrix[col, :] ^= matrix[row, :]
                cnot_list.append((row, col))
                diag_one = True
            matrix[row, :] ^= matrix[col, :]
            cnot_list.append((col, row))
        for row in checked_cols:
            if matrix[row, col] == 0:
                continue
            if not diag_one and row in col_span:
                raise Exception(f"Non-singular matrix: {matrix}")
            matrix[row, :] ^= matrix[col, :]
            cnot_list.append((col, row))
        checked_cols.add(col)

    return cnot_list


# GF(2) reduced row echelon form. Mutates matrix in place, returns it.
# Used by clean_matrix_col; does not record CNOT operations.
def gaussian_rref(matrix:np.ndarray):
    current_row = 0
    for c in range(matrix.shape[1]):
        if current_row >= matrix.shape[0]:
            break
        diag_one = False if matrix[current_row, c] == 0 else True
        for row in range(current_row+1, matrix.shape[0]):
            if matrix[row, c] == 0:
                continue
            if diag_one == False:
                matrix[current_row,:] ^= matrix[row,:]
                diag_one = True
            matrix[row,:] ^= matrix[current_row,:]
        if diag_one:
            for row in range(0, current_row):
                if matrix[row, c] == 0:
                    continue
                matrix[row,:] ^= matrix[current_row,:]
        current_row += 1
    return matrix

class MatrixException(Exception):
    ''''''

# Isolate output column `col` using only rows in row_span, via GF(2) RREF.
# Returns a CNOT list that makes out_matrix[:,col] a unit vector.
# Raises MatrixException if the column is not expressible from row_span.
def clean_matrix_col(phase_matrix:np.ndarray, out_matrix:np.ndarray, col:int, row_span:list[int]|None = None):
    if row_span == None:
        row_span = list(range(out_matrix.shape[0]))

    matrix = np.hstack((phase_matrix.copy(), out_matrix.copy()))[row_span, :]
    target_row = np.zeros(matrix.shape[1], int)
    target_row[col+phase_matrix.shape[1]] = 1

    rref = gaussian_rref(np.vstack([matrix.copy(), target_row]).T)
    cutoff_flag = False
    for r in rref:
        for i,v in enumerate(r):
            if i == rref.shape[1]-1:
                cutoff_flag = True
                if (v == 0):
                    break
            elif (v == 1):
                break
        else:
            raise MatrixException("Unsolvable Matrix")
        if cutoff_flag:
            break
    combo = list(map(lambda x : row_span[x], np.nonzero(rref[:,-1])[0]))
    isolated_column = out_matrix[:, col].copy().T

    cnot_list = []
    if col not in combo:
        cnot_list.append((combo[0], col))
        isolated_column[col] ^= isolated_column[combo[0]]

    for r in combo:
        if r == col:
            continue
        cnot_list.append((r, col))
        isolated_column[col] ^= isolated_column[r]

    for r2 in np.nonzero(isolated_column)[0]:
        if r2 == col:
            continue
        cnot_list.append((col, r2))
        isolated_column[r2] ^= isolated_column[col]

    return cnot_list


def synthesize_row_search(
    phasepoly:PhasePoly,
    buffer_size:int = -1,
    ends_checked:int = 10,
    dependencies:dict[int,set[int]] = dict(),
    child_buffer_size:int = -1,
    output_cost_model:Literal["gaussian", "parity_only"] = "gaussian",
    final_gaussian_elim_algorithm:Literal["modified", "classic"]|None = None,
):
    if output_cost_model not in ("gaussian", "parity_only"):
        raise ValueError(
            f"Invalid output_cost_model '{output_cost_model}'; "
            "must be 'gaussian' or 'parity_only'."
        )

    # Step 1: Build parity_table (n×R) and output_table (n×n) from the phase polynomial.
    # parity_table[:,k] = indicator vector of the k-th rotation's qubit support.
    # output_table[:,j] = target linear combination for output qubit j (identity = no permutation).
    parity_table = np.array([])
    output_table = np.array([])
    for _,s in phasepoly.rotations:
        col = np.array([(1 if q in s else 0) for q in phasepoly.qubits]).reshape(-1,1)
        if len(parity_table) > 0:
            parity_table = np.append(parity_table, col, 1)
        else:
            parity_table = col
    list_qubits = phasepoly.qubits
    list_output_qubits = [q for q in phasepoly.qubits if q in phasepoly.affineOut.keys()]
    for w in list_output_qubits:
        col = np.array([(1 if q in phasepoly.affineOut[w] else 0) for q in phasepoly.qubits]).reshape(-1,1)
        if len(output_table) > 0:
            output_table = np.append(output_table, col, 1)
        else:
            output_table = col

    # Drop qubit rows that carry no parity weight and no output requirement.
    ptr = 0
    list_used_qubits = list_qubits.copy()
    while ptr < len(list_used_qubits):
        if len(parity_table) > 0 and np.sum(parity_table, axis=1)[ptr] > 0:
            ptr += 1
            continue
        if np.sum(output_table, axis=1)[ptr] > 0:
            ptr += 1
            continue
        list_used_qubits.pop(ptr)
        if len(parity_table) > 0:
            parity_table = np.delete(parity_table, (ptr), axis=0)
        output_table = np.delete(output_table, (ptr), axis=0)

    # Translate qubit-id dependencies to row-index dependencies.
    # Rows in loc_dependencies are "locked" until their prerequisites are emitted
    # (used by multi-block synthesis to enforce block ordering).
    loc_dependencies:dict[int,set[int]] = dict()
    for qubit,depens in dependencies.items():
        loc_dependencies[list_used_qubits.index(qubit)] = set(list_used_qubits.index(q) for q in depens)
    valid_rows = [r for r in range(len(list_used_qubits)) if r not in loc_dependencies.keys()]

    # Mapping from row index → physical qubit id (used when emitting CX/Rz ops).
    mapping = utils.Mapping()
    for i,q in enumerate(list_used_qubits):
        mapping.map_qubit(i,q)

    # child_buffer_size caps the per-expansion child heap; keep it ≤ buffer_size.
    if child_buffer_size > buffer_size or child_buffer_size == -1 and buffer_size > 0:
        child_buffer_size = buffer_size

    @functools.total_ordering
    class State():
        def __init__(this, r_angles:list[float|utils.PiAngle], parity_table:np.ndarray, output_table:np.ndarray, 
                     active_cols:set[int], valid_rows:list[int], dependencies:dict[int,set[int]]):
            this.parity_table:np.ndarray = parity_table.copy()
            this.output_table:np.ndarray = output_table.copy()
            this.active_col:set[int] = active_cols
            this.past_ops = [] #full ops list from start to the prev state (1 state before)
            this.prev_ops = [] #diff from previous ops block
            this.ops_list_len:int = 0
            this.valid_rows:list[int] = valid_rows.copy() #rows not locked by dependencies
            this.dependencies:dict[int,set[int]] = dependencies #row prerequisite sets
            this.r_angles:list[float] = r_angles.copy()
            this._cost_val = None

        # Emit any rotation whose parity column has weight 1 (already on a single wire).
        def remove_ready_parities(this):
            this._cost_val = None

            if (
                len(this.parity_table) <= 0
                or this.parity_table.ndim < 2
                or this.parity_table.shape[1] <= 0
            ):
                this.active_col = set()
                return

            ready_parities = list(np.nonzero(np.sum(this.parity_table, 0) == 1)[0])
            offset = 0
            for col in sorted(ready_parities):
                row = np.nonzero(this.parity_table[:, col-offset])[0][0]
                if row not in this.valid_rows:
                    ready_parities.remove(col)
                    continue
                this.prev_ops.append(("rz",[mapping.get_physical_from_log(row)], [this.r_angles[col-offset]]))
                this.ops_list_len += 1
                this.parity_table = np.delete(this.parity_table, col-offset, 1)
                this.r_angles = np.delete(this.r_angles, col-offset)
                offset += 1
            this.active_col = set(c - len([None for p in ready_parities if p < c]) for c in this.active_col if c not in ready_parities)

        #Probably unused
        def redefine_active_cols(this):
            if len(this.active_col) == 0 and len(this.parity_table) > 0:
                if len(this.dependencies.keys()) == 0:
                    this.active_col = set(range(this.parity_table.shape[1]))
                else:
                    print("redefine active cols")
                    print(this.parity_table[sorted(this.dependencies.keys()), :])
                    this.active_col = set(np.nonzero(np.sum(this.parity_table[sorted(this.dependencies.keys()), :], 0) == 0)[0])
                    #TODO

        def cost(this):
            """
            Default cost tuple: (ops+parity+GE, parity, GE, -ops).
            parity_sum = total 1s remaining in parity_table (global work left).
            gaussian_len = GE CNOTs needed to realize current output_table (lookahead).
            Ordering by this tuple: prefer states that minimise total work while
            breaking ties in favour of fewer remaining parities, then cheaper GE,
            then more ops already done.

            ``parity_only`` mode is used by single_block_greedy: the search
            scores only the parity network, then appends output-basis GE after
            all rotations have been emitted.
            """
            if this._cost_val == None:
                ops_len = this.ops_list_len
                parity_sum = np.sum(this.parity_table)
                if output_cost_model == "parity_only":
                    this._cost_val = (ops_len + parity_sum, parity_sum, -ops_len)
                    return this._cost_val
                gaussian_len = 0
                try:
                    gaussian_len += len(gaussian_elim(this.output_table.copy(), this.valid_rows.copy()))
                except Exception:
                    gaussian_len += len(gaussian_elim(this.output_table.copy()))
                this._cost_val = (ops_len + parity_sum + gaussian_len, parity_sum, gaussian_len, -ops_len)
            return this._cost_val
        
        
        # Apply a CNOT list in dependency order, updating parity/output tables.
        def apply_cnot_list(this, cnots:list[tuple[int,int]]):
            this._cost_val = None

            node_degrees = defaultdict(int)

            for i,j in cnots:
                node_degrees[i] += 1
                node_degrees[j] += 1

            loc_cnots = cnots.copy()

            while len(loc_cnots) > 0:
                choices:set[int] = set()
                domain:set[int] = set(this.valid_rows)
                control_locked:set[int] = set()
                target_locked:set[int] = set()
                for i,t in enumerate(loc_cnots):
                    if t[0] not in control_locked and t[1] not in target_locked:
                        choices.add(i)
                    target_locked.add(t[0])
                    control_locked.add(t[1])
                    if domain.issubset(control_locked) and domain.issubset(target_locked):
                        break
                i,j = loc_cnots.pop(min(choices,
                    key=lambda x : 1))
                
                p_i = mapping.get_physical_from_log(i)
                p_j = mapping.get_physical_from_log(j)

                if len(this.parity_table) > 0:    
                    this.parity_table[j] ^= this.parity_table[i] 
                this.output_table[j] ^= this.output_table[i] 

                this.prev_ops.append(("cx", [p_j,p_i], []))
                this.ops_list_len += 1

                node_degrees[i] -= 1
                node_degrees[j] -= 1

        def full_ops_list(this):
            return this.past_ops + this.prev_ops

        def convert_to_output_state(this):
            this._cost_val = None
            this.apply_cnot_list(
                gaussian_elim(
                    this.output_table.copy(),
                    this.valid_rows,
                    algorithm=final_gaussian_elim_algorithm,
                )
            )

        # A pre-checking method to check if the state is linearly dependent.
        def linear_dependency_check(this, target_row_index: int) -> bool:
            """
            Check if the target row in the parity table is linearly independent of the other rows.
            Returns True if linearly independent (i.e., True means the state should be abandoned).
            """
            original_parity_table = this.parity_table.copy()
            original_output_table = this.output_table.copy()
            original_output_table = np.delete(original_output_table, [target_row_index], 1)
            original_parity_table = np.concatenate((original_parity_table, original_output_table), 1)
            target_row = original_parity_table[target_row_index]
            modified_parity_table = original_parity_table[[i for i in this.valid_rows if i != target_row_index], :]
            # Full-rank check over GF(2): if the remaining rows span the space,
            # the target row is linearly dependent and this state can be abandoned.
            out = np.linalg.matrix_rank(modified_parity_table % 2) == len(this.valid_rows)
            return out

        def get_expired_rows(this):
            return set(range(parity_table.shape[0])) - set(this.valid_rows) - this.dependencies.keys()

        def __eq__(this, other:Self):
            return this.cost() == other.cost()
        def __lt__(this, other:Self):
            return this.cost() < other.cost()


    # Step 2: A*-style parity-network search.
    # States are (parity_table, output_table) snapshots after a sequence of CNOTs.
    # The DEPQ keeps up to buffer_size states; poplast() extracts the lowest-cost one.
    initial_state = State([r[0] for r in phasepoly.rotations], parity_table, output_table, set(), valid_rows, loc_dependencies)
    initial_state.remove_ready_parities()

    buffer:depq.DEPQ = depq.DEPQ()
    end_states:list[State] = []
    buffer.insert(initial_state, initial_state.cost())

    def add_state_to_queue(new_state:State, queue:depq.DEPQ, queue_cap:int = -1):
        try:
            new_state.cost()
        except Exception:
            return

        # Reject states where a parity column has support on a locked row that is
        # not yet reachable (would require a CNOT on a qubit not yet "unlocked").
        if len(new_state.parity_table) > 0 and new_state.parity_table.shape[1] > 0:
            column_parities = [np.nonzero(new_state.parity_table[:,c].reshape(-1))[0] for c in range(new_state.parity_table.shape[1])]
            if not all((r in new_state.valid_rows or r in new_state.dependencies.keys()) for col in column_parities for r in col):
                return
        queue.insert(new_state, new_state.cost())
        if queue.size() > queue_cap and queue_cap != -1:
            queue.popfirst()

    # Step 2.1: Main search loop.
    while not buffer.is_empty():

        childbuffer:depq.DEPQ = depq.DEPQ()
        state:State = buffer.poplast()[0]

        #suppress type errors
        if not isinstance(state, State):
            raise Exception()

        active_cols = state.active_col
        if len(state.active_col) == 0 and len(state.parity_table) > 0:
            if len(dependencies.keys()) == 0:
                active_cols = set(range(state.parity_table.shape[1]))
            else:
                # Only columns with no support on locked rows are reachable now.
                active_cols = set(np.nonzero(np.sum(state.parity_table[sorted(state.dependencies.keys()), :], 0) == 0)[0])
                #TODO

        # Terminal state: all rotations emitted; finalize output linear function via GE.
        if len(state.parity_table) <= 0 or state.parity_table.shape[1] <= 0:
            if len(state.dependencies) == 0:
                try:
                    state.convert_to_output_state()
                except:
                    continue
                end_states.append(state)
                if len(end_states) >= ends_checked:
                    break
                continue

        # Step 2.1.1: Expand the current state.
        if len(active_cols) > 0:
            # Normal expansion: try every (i→j) CX that reduces at least one active column.
            active_rows = np.nonzero(np.sum(state.parity_table[:,list(sorted(active_cols))], 1) > 0)[0]
            active_rows = [r for r in active_rows if r in state.valid_rows]

            any_possible_cnots = False
            for i in active_rows:
                for j in active_rows:
                    if i == j:
                        continue
                    # Only consider CX(j→i) if it reduces ≥1 active column.
                    helped_cols = set(c for c in active_cols if state.parity_table[i,c] == 1 and state.parity_table[j,c] == 1)
                    if len(helped_cols) == 0:
                        continue

                    any_possible_cnots = True
                    new_pt = state.parity_table.copy()
                    new_pt[i] ^= new_pt[j]
                    new_ot = state.output_table.copy()
                    new_ot[i] ^= new_ot[j]

                    new_state = State(state.r_angles, new_pt, new_ot, helped_cols, state.valid_rows, state.dependencies)
                    new_state.past_ops = state.full_ops_list()
                    new_state.prev_ops.insert(0, ("cx", [mapping.get_physical_from_log(i),mapping.get_physical_from_log(j)], []))
                    new_state.ops_list_len += state.ops_list_len + 1
                    new_state.remove_ready_parities()
                    add_state_to_queue(new_state, childbuffer, child_buffer_size)

            finished_rows = np.nonzero(np.sum(state.parity_table, 1).reshape(-1) == 0)[0]
        else:
            # No active columns: all remaining rotations are blocked by row dependencies.
            # Attempt to unlock a dependency by isolating that row in the output table.
            if all([state.linear_dependency_check(i) for i in state.valid_rows if any(i in j for j in state.dependencies.values())]):
                print("lindep check fail!!")
                break

            rowcheck = [i for i in state.valid_rows if any(i in j for j in state.dependencies.values()) and not state.linear_dependency_check(i)]

            # Eliminate rows for row dependencies (multi-block synthesis).
            for row in rowcheck:
                if row not in state.valid_rows or not any([row in s for s in state.dependencies.values()]):
                    continue

                new_pt = state.parity_table.copy()
                new_ot = state.output_table.copy()

                try:
                    op_list = clean_matrix_col(new_pt.copy(), new_ot.copy(), row, state.valid_rows)
                except MatrixException:
                    continue
                new_vr = state.valid_rows.copy()
                new_vr.remove(row)

                new_depens = copy.deepcopy(state.dependencies)
                remo_set = set()
                for new_r in new_depens.keys():
                    s = new_depens[new_r]
                    if row not in s:
                        continue
                    s.remove(row)
                    new_depens[new_r] = s
                    if len(s) == 0:
                        new_vr.append(new_r)
                        remo_set.add(new_r)
                for remo_row in remo_set:
                    new_depens.pop(remo_row)

                new_state = State(state.r_angles, new_pt, new_ot, active_cols, new_vr, new_depens)
                new_state.past_ops = state.full_ops_list()
                new_state.ops_list_len += state.ops_list_len
                new_state.remove_ready_parities()
                new_state.apply_cnot_list(op_list)
                add_state_to_queue(new_state, childbuffer, child_buffer_size)

        while childbuffer.size() > 0:
            child_out:State = childbuffer.poplast()[0]
            add_state_to_queue(child_out, buffer, buffer_size)

    end_states.sort()
    out:Circuit = Circuit(qubits=phasepoly.qubits)
    if len(end_states) <= 0:
        raise utils.NoResultsError(f"No Final State Found: for {phasepoly}")
    final_state = end_states[0]
    for g in final_state.full_ops_list():
        out.append_node(Gate(g[0], g[1], g[2]))
    return out


def synthesize_single_block_greedy(phasepoly:PhasePoly):
    return synthesize_row_search(
        phasepoly,
        buffer_size=1,
        ends_checked=1,
        child_buffer_size=1,
        output_cost_model="parity_only",
        final_gaussian_elim_algorithm="modified",
    )


def synthesize_single_block_greedy_classical_GE(phasepoly:PhasePoly):
    return synthesize_row_search(
        phasepoly,
        buffer_size=1,
        ends_checked=1,
        child_buffer_size=1,
        output_cost_model="parity_only",
        final_gaussian_elim_algorithm="classic",
    )
        

def __insert_cnot(out_circ:Circuit, i,j, p_table, o_table):
    out_circ.append_node(Gate("cx", [i,j]))
    if len(p_table) > 0:
        p_table[i] ^= p_table[j]
    o_table[i] ^= o_table[j]

@functools.total_ordering
class QueueObject():
    def __init__(this):
        this.parity_table:np.ndarray
        this.output_table:np.ndarray
        this.ops_list:list[tuple[int,int]]
        this._cost_val = None

    def cost(this):
        # Cost = CNOTs applied so far + GE cost to realize the remaining output linear function.
        if this._cost_val == None:
            this._cost_val = len(this.ops_list) + len(gaussian_elim(this.output_table.copy()))
        return this._cost_val

    def __eq__(this, other:Self):
        return this.cost() == other.cost()
    def __lt__(this, other:Self):
        return this.cost() < other.cost()

# Synthesize a circuit equivalent to phasepoly using greedy column-by-column reduction.
# Outer loop: pick the minimum-Hamming-weight column (rotation); inner BFS finds
# the cheapest CNOT sequence to reduce that column to a single 1 (so the Rz can fire).
# After all rotations are placed, Gaussian elimination realizes the output linear function.
def synthesize_by_col(phasepoly:PhasePoly):
    out = Circuit(qubits=phasepoly.qubits)
    num_qubits = len(phasepoly.qubits)

    r_angles = [r[0] for r in phasepoly.rotations]
    parity_table = np.array([])
    output_table = np.identity(num_qubits, dtype=int)
    for _,s in phasepoly.rotations:
        col = np.array([(1 if q in s else 0) for q in phasepoly.qubits]).reshape(-1,1)
        if len(parity_table) > 0:
            parity_table = np.append(parity_table, col, 1)
        else:
            parity_table = col

    list_qubits = phasepoly.qubits
    list_output_qubits = [q for q in phasepoly.qubits if q in phasepoly.affineOut.keys()]

    for i,w in enumerate(phasepoly.qubits):
        if w not in phasepoly.affineOut.keys():
            continue
        col = np.array([(1 if q in phasepoly.affineOut[w] else 0) for q in phasepoly.qubits])
        output_table[:,i] = col

    ptr = 0
    list_used_qubits = list_qubits.copy()
    while ptr < len(list_used_qubits):
        if len(parity_table) > 0 and np.sum(parity_table, axis=1)[ptr] > 0:
            ptr += 1
            continue
        if np.sum(output_table, axis=1)[ptr] > 0:
            ptr += 1
            continue
        list_used_qubits.pop(ptr)
        if len(parity_table) > 0:
            parity_table = np.delete(parity_table, (ptr), axis=0)
        output_table = np.delete(output_table, (ptr), axis=0)
        output_table = np.delete(output_table, (ptr), axis=1)

    weights = np.identity(num_qubits)

    def implement_rotation(col):
        nonlocal parity_table, r_angles, out, weights
        
        if sum(parity_table[:,col]) != 1:
            buffer:list[QueueObject] = []
            item = QueueObject()
            item.parity_table = parity_table.copy()
            item.output_table = output_table.copy()
            item.ops_list = []
            heapq.heappush(buffer, item)

            item:QueueObject
            while True:
                item = heapq.heappop(buffer)
                lines = np.where(item.parity_table[:, col])[0]
                if len(lines) == 1:
                    break
                for i in lines:
                    for j in lines:
                        if i == j:
                            continue
                        new_item = QueueObject()
                        new_item.parity_table = item.parity_table.copy()
                        new_item.parity_table[i] ^= new_item.parity_table[j]
                        new_item.output_table = item.output_table.copy()
                        new_item.output_table[i] ^= new_item.output_table[j]
                        new_item.ops_list = item.ops_list.copy()
                        new_item.ops_list.append((i,j))
                        heapq.heappush(buffer, new_item)
            
            for (i, j) in item.ops_list:
                __insert_cnot(out, i, j, parity_table, output_table)
        index = np.nonzero(parity_table[:,col])[0][0]
        out.append_node(Gate("rz", [index], [r_angles[col]])) 
        parity_table = np.delete(parity_table, col, 1)
        r_angles = np.delete(r_angles, col)

    # Greedily pick columns with minimum Hamming weight, breaking ties by qubit order.
    while len(parity_table) > 0 and parity_table.shape[1]:
        cols = np.argwhere(sum(parity_table) == np.amin(sum(parity_table)))
        for i,_ in enumerate(phasepoly.qubits):
            ones = cols[np.where(parity_table[i, cols])[0]]
            if len(ones):
                cols = ones
                if len(cols) == 1:
                    break
        implement_rotation(cols[0][0])

    # Emit final CNOT cascade to realize the output linear function via GE.
    circuit_l = gaussian_elim(output_table.copy(), list(range(num_qubits)), col_span=[i for i,q in enumerate(phasepoly.qubits) if q in list_output_qubits])
    for i,j in circuit_l:
        out.append_node(Gate("cx", [phasepoly.qubits[j],phasepoly.qubits[i]]))

    return out

#synthesize, but the results replace the current circuit in place
def synthesize_in_place(phasepoly:PhasePoly, synthesis_type:Literal["single_block_greedy", "single_block_greedy_classical_GE", "row_heap", "revert"],
     buffer_size:int = -1, ends_checked:int = 10, only_replace_on_improve:bool = True, child_buffer_size:int = -1):
    result:Circuit

    if synthesis_type == "single_block_greedy":
        result = synthesize_single_block_greedy(phasepoly)
    elif synthesis_type == "single_block_greedy_classical_GE":
        result = synthesize_single_block_greedy_classical_GE(phasepoly)
    elif synthesis_type == "row_heap":
        result = synthesize_row_search(phasepoly, buffer_size, ends_checked, child_buffer_size=child_buffer_size)
    elif synthesis_type == "revert":
        if len(phasepoly.possible_circuits) < 1:
            #raise Exception("No Phasepoly circuit to revert to")
            result = synthesize_single_block_greedy(phasepoly)
            c = CommentLine(phasepoly.qubits, "Phasepoly could not be reverted, single_block_greedy used instead.")
            c.attach_before(dict([(q,n.next(q)) for q,n in result.ins.items()]))
        else:
            result = phasepoly.possible_circuits[0]
    else:
        raise ValueError(
            f"Invalid synthesis_type '{synthesis_type}'; "
            "must be 'single_block_greedy', 'single_block_greedy_classical_GE', "
            "'row_heap', or 'revert'."
        )

    if only_replace_on_improve:
        res_seq = result.get_sequence()
        res_dat = (len([True for g in res_seq if g.nodeType == 'cx']), len(res_seq))
        for circ in phasepoly.possible_circuits:
            if len(circ.qubits) > len(result.qubits):
                continue
            circ_seq = circ.get_sequence()
            if res_dat > (len([True for g in circ_seq if g.nodeType == 'cx'])+len([True for g in circ_seq if g.nodeType == 'swap'])*3, len(circ_seq)):
                result = circ

    if result.qubits != phasepoly.qubits and all(c in result.qubits for c in phasepoly.qubits):
        pass#TODO

    result.replace_at(Circuit(ins=phasepoly.ins, outs=phasepoly.outs))


#Call AFTER partitioning
def get_phasePoly_groups(circuit:Circuit, maxsize:int = -1, metric:Literal['phasepoly_count','qubit_rotation_sum'] = 'phasepoly_count'):
    def cost(group:list[tuple[int, PhasePoly]]):
        if metric == 'qubit_rotation_sum':
            sum = 0
            for _,p in group:
                sum += len(p.qubits)*len(p.rotations)
            return sum
        return len(group)
    
    out:list[list[tuple[int,PhasePoly]]] = []
    for seq_no,p in enumerate(circuit.get_sequence()):
        if not isinstance(p, PhasePoly):
            continue

        current_group_index = None
        j = 0
        while j <= len(out):
            if j == len(out):
                if current_group_index == None:
                    out.append([(seq_no,p)])
                break
            s = out[j]
            if maxsize != -1 and cost(s) >= maxsize:
                j += 1
                continue
            for _,p2 in s:
                if len([True for v in p2.outs.values() if p == v]) > 1:
                    if current_group_index == None:
                        if maxsize != -1 and cost(s + [(seq_no,p)]) > maxsize:
                            continue
                        s.append((seq_no,p))
                        current_group_index = j
                    elif maxsize == -1 or cost(out[j] + out[current_group_index]) <= maxsize:
                        out[current_group_index].extend(out[j])
                        out.pop(j)
                        j -= 1
                    break  
            j += 1

    out_real = [[t[1] for t in sorted(s, key=lambda x : x[0])] for s in out]
    return out_real

READABLE_FAKE_QUBITS = False

#WIP, use with caution. Do not use on physical circuits.
def synthesize_grouped_phasePoly_in_place(phasePolys:list[PhasePoly], buffer_size:int = -1, ends_checked:int = 10, only_replace_on_improve:bool = True, greater_circuit:Circuit = None, child_buffer_size=-1):
    
    old_cnot_count = None
    if only_replace_on_improve and min([len(pp.possible_circuits) for pp in phasePolys]) > 0:
        old_cnot_count = sum([min([len([True for n in pc.get_sequence() if n.nodeType == "cx"]) for pc in pp.possible_circuits]) for pp in phasePolys])
    current_mapping:dict[int,list[int|float]] = dict()
    rev_map = dict()
    qubit_location:dict[int|float,PhasePoly] = dict()

    combined_phasePoly = PhasePoly([], [], dict())
    mapped_poly_qubits:list[list[int]] = []
    fake_qubit_num = 0.5
    for poly in phasePolys:
        input_parities = []

        poly_in_context = poly.copy_disconnected()

        for q in poly.qubits:
            if q not in current_mapping.keys():
                current_mapping[q] = [q]
                combined_phasePoly.qubits.append(q)
                combined_phasePoly.affineOut[q] = {q}
                rev_map[q] = q
                qubit_location[q] = poly
                continue
            if poly.prev(q) not in phasePolys:
                # q's input comes from outside the group: allocate a fake qubit to
                # represent the post-previous-block value of q in the fused frame.
                fake_qubit = fake_qubit_num
                fake_qubit_num += 1

                current_mapping[q].append(fake_qubit)
                rev_map[fake_qubit] = q
                qubit_location[fake_qubit] = poly
                combined_phasePoly.qubits.append(fake_qubit)
                combined_phasePoly.affineOut[fake_qubit] = {fake_qubit}

            if q != current_mapping[q][-1]:
                poly_in_context = poly_in_context.replace_qubit(q, current_mapping[q][-1])
        input_parities.extend([(q, combined_phasePoly.affineOut[q]) for q in poly_in_context.qubits])
        poly_in_context = poly_in_context.remap_inputs_change_parities(dict(input_parities))
        mapped_poly_qubits.append(poly_in_context.qubits)

        for theta,parity in poly_in_context.rotations:
            for i in range(len(combined_phasePoly.rotations)):
                if combined_phasePoly.rotations[i][1] == parity:
                    combined_phasePoly.rotations[i] = (combined_phasePoly.rotations[i][0]+theta, parity)
                    break
            else:
                combined_phasePoly.rotations.append((theta, parity))
        for q,out_parity in poly_in_context.affineOut.items():
            combined_phasePoly.affineOut[q] = out_parity    
    # Drop rotations whose angles cancelled to zero across blocks.
    combined_phasePoly.rotations = [t for t in combined_phasePoly.rotations if t[0] != 0.0]

    # Build dependency graph: fake_qubit depends on the real/fake qubit it was split from,
    # plus any cross-block flow reachable via greater_circuit's neighbourhood.
    if greater_circuit != None:
        seq_nums = dict((b,a) for a,b in enumerate(greater_circuit.get_sequence()))

    dependencies:dict[int,set[int]] = dict()
    for qubit_order in current_mapping.values():
        for i in range(len(qubit_order)-1):
            if dependencies.get(qubit_order[i+1]) == None:
                dependencies[qubit_order[i+1]] = set()
            dependencies[qubit_order[i+1]].add(qubit_order[i])
    
    if greater_circuit != None:
        last_seq = max(seq_nums[pp] for pp in phasePolys)
    
    for pp in phasePolys:
        for q in pp.qubits:
            if pp.next(q) in phasePolys:
                continue
            converted_q_in = mapped_poly_qubits[phasePolys.index(pp)][pp.qubits.index(q)]
            
            buffer:deque[tuple[int,Node]] = deque()
            tracked = set()
            buffer.append((q,pp.next(q)))
            while len(buffer) > 0:
                new_q, new_node = buffer.popleft()
                if new_node in phasePolys:
                    converted_q_out = mapped_poly_qubits[phasePolys.index(new_node)][new_node.qubits.index(new_q)]
                    if dependencies.get(converted_q_out) == None:
                        dependencies[converted_q_out] = set()
                    dependencies[converted_q_out].add(converted_q_in)
                    continue

                if new_node in tracked:
                    continue
                tracked.add(new_node)
                if greater_circuit != None:
                    if new_node not in seq_nums or seq_nums[new_node] > last_seq:
                        continue

                for more_q, more_node in new_node.outs.items():
                    buffer.append((more_q, more_node))
    
    try:
        new_combined_circ:Circuit = synthesize_row_search(combined_phasePoly, buffer_size, ends_checked=ends_checked, dependencies=dependencies, child_buffer_size=child_buffer_size)
    except Exception as e:
        print("\nNOTE: resorted non non-merge methods")
        print(e)
        raise e

    ncc_seq = new_combined_circ.get_sequence()
    new_cnot_count = len([True for n in ncc_seq if n.nodeType == "cx"])
    new_cnot_count += len([True for n in ncc_seq if n.nodeType == "swap"])*3

    # Revert to per-block synthesis if the fused result is worse (can happen when
    # the combined parity table is wider and the heap explores suboptimal choices).
    if old_cnot_count != None and old_cnot_count < new_cnot_count:
        for poly in phasePolys:
            synthesize_in_place(poly, "row_heap", buffer_size, ends_checked, only_replace_on_improve)
        return

    # Splice synthesized gates back into the host circuit, translating fake qubit IDs
    # back to real ones via rev_map.
    for gate in new_combined_circ.get_sequence():
        if gate.nodeType == "commentLine":
            continue
        new_gate = gate.copy_disconnected()
        new_gate.qubits = [rev_map[q] for q in gate.qubits]
        new_gate.attach_before(dict([(rev_map[q], qubit_location[q]) for q in gate.qubits]))

    for poly in phasePolys:
        poly.remove_from_circuit()

    pass

#!/usr/bin/env python3
# src.quantum_layouts
"""
Quantum Device Layout Generator (networkx)

Supports:
  - Grid: make_grid_layout(rows, cols) — arbitrary rows x cols 4-neighbor grid
  - IBM Heavy-Hex: make_ibm_heavyhex_layout('falcon'|'eagle') from real coupling map
  - Custom Heavy-Hex: make_heavyhex_custom(hex_rows, hex_cols) — N = hex_rows * hex_cols hex cells

Notes:
  - IBM layouts are built from *real coupling maps*:
      (1) Prefer local qiskit_ibm_runtime fake backends (no network dependency)
      (2) Else fetch backend conf JSON from GitHub and parse "coupling_map"
      (3) Else fail loudly (do NOT fabricate an approximate heavy-hex)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Optional
import json
import os
import hashlib
import networkx as nx

try:
    import urllib.request
    HAS_URLLIB = True
except Exception:
    HAS_URLLIB = False


# =============================================================================
# 1) GRID (parameterized: rows x cols)
# =============================================================================

def make_grid_layout(rows: int, cols: Optional[int] = None) -> nx.Graph:
    """
    Build a 4-neighbor rectangular grid coupling graph.

    Args:
        rows: Number of rows.
        cols: Number of columns; if None, use cols=rows (square grid).

    Returns:
        networkx.Graph with rows*cols nodes; nodes have attributes row, col, coord.
    """
    if rows < 1 or (cols is not None and cols < 1):
        raise ValueError("rows and cols must be >= 1")
    c = cols if cols is not None else rows
    G = nx.Graph(name=f"grid_{rows}x{c}")

    def nid(r: int, col: int) -> int:
        return r * c + col

    for r in range(rows):
        for col in range(c):
            u = nid(r, col)
            G.add_node(u, row=r, col=col, coord=(r, col))
            if r + 1 < rows:
                G.add_edge(u, nid(r + 1, col))
            if col + 1 < c:
                G.add_edge(u, nid(r, col + 1))

    return G


# =============================================================================
# 2) IBM HEAVY-HEX (27/127) via real coupling map
# =============================================================================

@dataclass(frozen=True)
class IBMBackendSpec:
    key: str
    n_qubits: int
    conf_url: str


IBM_SPECS: Dict[str, IBMBackendSpec] = {
    # 27-qubit heavy-hex (Falcon-family)
    "falcon": IBMBackendSpec(
        key="falcon",
        n_qubits=27,
        conf_url="https://raw.githubusercontent.com/Qiskit/qiskit-ibm-runtime/main/"
                 "qiskit_ibm_runtime/fake_provider/backends/kolkata/conf_kolkata.json",
    ),
    # 127-qubit heavy-hex (Eagle-family)
    "eagle": IBMBackendSpec(
        key="eagle",
        n_qubits=127,
        conf_url="https://raw.githubusercontent.com/Qiskit/qiskit-ibm-runtime/main/"
                 "qiskit_ibm_runtime/fake_provider/backends/washington/conf_washington.json",
    ),
    # 133-qubit heavy-hex (Heron-family)
    "heron": IBMBackendSpec(
        key="heron",
        n_qubits=156,
        conf_url="https://raw.githubusercontent.com/Qiskit/qiskit-ibm-runtime/main/"
                 "qiskit_ibm_runtime/fake_provider/backends/fez/conf_fez.json",
    ),
}

# Cache directory for downloaded JSON files
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".ibm_backend_cache")


def _get_cache_path(url: str) -> str:
    """Get cache file path for a given URL."""
    # Create cache directory if it doesn't exist
    os.makedirs(_CACHE_DIR, exist_ok=True)
    
    # Use URL hash as filename to avoid path issues
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(_CACHE_DIR, f"{url_hash}.json")


def _load_json_from_cache(url: str) -> Optional[dict]:
    """Load JSON from local cache if available."""
    cache_path = _get_cache_path(url)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            # If cache file is corrupted, remove it
            try:
                os.remove(cache_path)
            except Exception:
                pass
    return None


def _save_json_to_cache(url: str, data: dict) -> None:
    """Save JSON data to local cache."""
    try:
        cache_path = _get_cache_path(url)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        # If caching fails, just continue without caching
        pass


def _fetch_json(url: str, timeout_s: int = 20, use_cache: bool = True) -> dict:
    """
    Fetch JSON over HTTP(S) with local caching.
    
    Args:
        url: URL to fetch JSON from
        timeout_s: Timeout in seconds
        use_cache: If True, use local cache if available, and save to cache after fetching
    
    Returns:
        Parsed JSON data as dict
    """
    # Try to load from cache first
    if use_cache:
        cached_data = _load_json_from_cache(url)
        if cached_data is not None:
            return cached_data
    
    # Fetch from network
    if not HAS_URLLIB:
        raise RuntimeError("urllib not available")
    
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    # Save to cache for future use
    if use_cache:
        _save_json_to_cache(url, data)
    
    return data


def _build_undirected_graph_from_coupling_map(
    n_qubits: int,
    coupling_map: Sequence[Sequence[int]],
) -> nx.Graph:
    """Build an undirected connectivity graph from a directed coupling map list."""
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    for e in coupling_map:
        if len(e) != 2:
            continue
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        G.add_edge(u, v)
    return G


def make_ibm_heavyhex_layout(which: str, prefer_local_fake_backend: bool = True) -> nx.Graph:
    """
    Return IBM heavy-hex connectivity graph for:
      - 'falcon' (27)
      - 'eagle'  (127)

    Priority:
      1) If qiskit_ibm_runtime is installed: use FakeKolkataV2 / FakeWashingtonV2
      2) Else: fetch backend conf JSON from GitHub and parse coupling_map
      3) Else: raise (do NOT fabricate an approximate heavy-hex)
    """
    key = which.strip().lower()
    if key not in IBM_SPECS:
        raise ValueError(f"which must be one of {sorted(IBM_SPECS)}, got {which}")

    spec = IBM_SPECS[key]

    # 1) Prefer local fake backend (no network)
    if prefer_local_fake_backend:
        try:
            if key == "falcon":
                from qiskit_ibm_runtime.fake_provider import FakeKolkataV2  # type: ignore
                backend = FakeKolkataV2()
            else:
                from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2  # type: ignore
                backend = FakeWashingtonV2()

            cm = backend.coupling_map
            edges = cm.get_edges()

            G = nx.Graph(name=f"ibm_{key}_{spec.n_qubits}_local_fake")
            G.add_nodes_from(range(spec.n_qubits))
            G.add_edges_from(edges)
            G.graph["source"] = f"qiskit_ibm_runtime.fake_provider.{backend.__class__.__name__}"
            return G
        except Exception:
            pass

    # 2) Fetch config from GitHub
    last_err: Optional[Exception] = None
    try:
        conf = _fetch_json(spec.conf_url)
        coupling_map = conf.get("coupling_map")
        if not coupling_map:
            raise RuntimeError("No coupling_map found in backend config JSON")

        G = _build_undirected_graph_from_coupling_map(spec.n_qubits, coupling_map)
        G.graph["name"] = f"ibm_{key}_{spec.n_qubits}_from_conf"
        G.graph["source"] = spec.conf_url
        return G
    except Exception as e:
        last_err = e

    # 3) Fail loudly (no approximation)
    raise RuntimeError(
        f"Cannot build IBM {key} heavy-hex layout. "
        f"Tried local fake backend and URL fetch; last error: {last_err}"
    )


# =============================================================================
# 3) Custom HEAVY-HEX (parameterized by number of hex cells)
# =============================================================================

def make_heavyhex_custom(hex_rows: int, hex_cols: int) -> nx.Graph:
    """
    Build a heavy-hex coupling graph with N = hex_rows * hex_cols hex cells
    (honeycomb + heavy nodes on edges, same topology as IBM heavy-hex).

    Structure: each hex cell has 6 edges; each edge has one heavy qubit (degree 2);
    hex vertices are qubits (degree 2 or 3). Adjacent hexagons share edges and endpoints.

    Args:
        hex_rows: Number of hex cell rows.
        hex_cols: Number of hex cell columns.

    Returns:
        networkx.Graph with nodes 0..n-1; max degree 3 in ideal case (may be higher at boundary).
    """
    if hex_rows < 1 or hex_cols < 1:
        raise ValueError("hex_rows and hex_cols must be >= 1")

    # Pointy-top hex integer vertex coords (2q+r, 3r); hex (row,col) -> axial (q,r)=(col,row).
    # Six corners: 0:(2q+r,3r), 1:(2q+r+2,3r), 2:(2q+r+1,3r+3), 3:(2q+r-1,3r+3), 4:(2q+r-2,3r), 5:(2q+r-1,3r-3)
    def vertex_keys(q: int, r: int) -> list[tuple[int, int]]:
        return [
            (2 * q + r, 3 * r),
            (2 * q + r + 2, 3 * r),
            (2 * q + r + 1, 3 * r + 3),
            (2 * q + r - 1, 3 * r + 3),
            (2 * q + r - 2, 3 * r),
            (2 * q + r - 1, 3 * r - 3),
        ]

    vertices: set[tuple[int, int]] = set()
    edges_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for row in range(hex_rows):
        for col in range(hex_cols):
            q, r = col, row
            keys = vertex_keys(q, r)
            for k in keys:
                vertices.add(k)
            for i in range(6):
                v1, v2 = keys[i], keys[(i + 1) % 6]
                e = (min(v1, v2), max(v1, v2))
                edges_set.add(e)

    # Vertex -> node id; edge -> node id (appended after vertices)
    vlist = sorted(vertices)
    vertex_to_id = {v: i for i, v in enumerate(vlist)}
    n_vertices = len(vlist)
    elist = sorted(edges_set)
    edge_to_id = {e: n_vertices + i for i, e in enumerate(elist)}

    G = nx.Graph(name=f"heavyhex_{hex_rows}x{hex_cols}")
    G.add_nodes_from(range(n_vertices + len(elist)))
    G.graph["hex_rows"] = hex_rows
    G.graph["hex_cols"] = hex_cols
    G.graph["n_hex_cells"] = hex_rows * hex_cols

    for (v1, v2) in elist:
        ev = edge_to_id[(v1, v2)]
        G.add_edge(vertex_to_id[v1], ev)
        G.add_edge(vertex_to_id[v2], ev)

    # Geometric positions for terminal/hex layout: vertex at 2x (2q+r,3r), edge at midpoint
    for i, v in enumerate(vlist):
        G.nodes[i]["pos"] = (4 * v[0], 2 * v[1])
    for (v1, v2) in elist:
        ev = edge_to_id[(v1, v2)]
        G.nodes[ev]["pos"] = (2 * (v1[0] + v2[0]), v1[1] + v2[1])

    return G


def _heavyhex_terminal_grid(G: nx.Graph, cell_w: int = 2) -> list[list[str]]:
    """Draw heavy-hex on a 2D character grid using geometric positions. cell_w: columns per unit (for diagonals)."""
    pos = {}
    for n in G.nodes():
        if "pos" in G.nodes[n]:
            pos[n] = G.nodes[n]["pos"]
    if not pos or len(pos) != G.number_of_nodes():
        return [["(no layout positions)"]]

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Grid: row = y (top to bottom), col = x; each x unit uses cell_w columns
    width = (max_x - min_x + 1) * cell_w
    height = max_y - min_y + 1
    grid = [[" "] * (width + 1) for _ in range(height)]

    def to_col(x: int) -> int:
        return min((x - min_x) * cell_w, width - 1)

    def to_row(y: int) -> int:
        return y - min_y

    # Draw edges first (dense sampling; use - | / \ by direction)
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        steps = max(abs(dx), abs(dy), 1) * (cell_w + 1)
        for k in range(steps + 1):
            t = k / steps
            x = x1 + t * dx
            y = y1 + t * dy
            r, c = to_row(round(y)), to_col(round(x))
            if 0 <= r < height and 0 <= c < width:
                if grid[r][c] == " ":
                    if abs(dy) < 0.01:
                        grid[r][c] = "-"
                    elif abs(dx) < 0.01:
                        grid[r][c] = "|"
                    elif dx * dy > 0:
                        grid[r][c] = "\\"
                    else:
                        grid[r][c] = "/"

    # Draw nodes (overwrite with 'o')
    for n in G.nodes():
        x, y = pos[n]
        r, c = to_row(y), to_col(x)
        if 0 <= r < height and 0 <= c < width:
            grid[r][c] = "o"

    return grid


def print_heavyhex_terminal(
    hex_rows: int = 2,
    hex_cols: int = 2,
    cell_width: int = 2,
    title: bool = True,
) -> str:
    """Print heavy-hex as ASCII art in the terminal (hex geometric layout). Returns the string for redirection/testing."""
    G = make_heavyhex_custom(hex_rows, hex_cols)
    grid = _heavyhex_terminal_grid(G, cell_w=cell_width)
    lines = []
    if title:
        lines.append(f" Heavy-Hex {hex_rows}x{hex_cols} ({hex_rows * hex_cols} hex cells) — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ")
    for row in grid:
        lines.append("".join(row).rstrip())
    text = "\n".join(lines)
    print(text)
    return text


# =============================================================================
# Minimal sanity tests
# =============================================================================

def _sanity():
    for rows, cols in ((4, 4), (5, 5), (6, 6)):
        G = make_grid_layout(rows, cols)
        assert G.number_of_nodes() == rows * cols

    for which, n in (("falcon", 27), ("eagle", 127)):
        G = make_ibm_heavyhex_layout(which)
        assert G.number_of_nodes() == n
        assert nx.is_connected(G)
        assert max(dict(G.degree()).values()) <= 3  # heavy-hex degree constraint

    # Custom heavy-hex
    G = make_heavyhex_custom(2, 2)
    assert nx.is_connected(G)

    for which, n in (("falcon", 27), ("eagle", 127)):
        G = make_ibm_heavyhex_layout(which)
        assert G.number_of_nodes() == n
        assert nx.is_connected(G)
        assert max(dict(G.degree()).values()) <= 3  # heavy-hex degree constraint


# ============================================================================
# 4. TESTING
# ============================================================================

def test_all_layouts():
    """Test all layout generators."""
    print("=" * 60)
    print("Layout Generation Test Suite")
    print("=" * 60)

    # Test grids (parameterized rows x cols)
    print("\nTesting Grid Layouts:")
    for rows, cols in [(4, 4), (5, 5), (3, 6)]:
        G = make_grid_layout(rows, cols)
        n_qubits = rows * cols
        expected_edges = (rows - 1) * cols + rows * (cols - 1) if (rows > 1 or cols > 1) else 0
        assert G.number_of_nodes() == n_qubits
        assert G.number_of_edges() == expected_edges
        assert all("row" in G.nodes[n] and "col" in G.nodes[n] for n in G.nodes())
        print(f"  ✓ Grid {rows}×{cols} ({n_qubits} qubits): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Test IBM Heavy-Hex
    print("\nTesting IBM Heavy-Hex Layouts:")
    for which in ["falcon", "eagle"]:
        expected_nodes = 27 if which == "falcon" else 127
        G = make_ibm_heavyhex_layout(which, prefer_local_fake_backend=True)
        assert G.number_of_nodes() == expected_nodes
        source = G.graph.get("source", "unknown")
        print(f"  ✓ {which.capitalize()}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"    Source: {source}")

    # Test custom Heavy-Hex
    print("\nTesting Custom Heavy-Hex Layouts:")
    for hrows, hcols in [(2, 2), (2, 3)]:
        G = make_heavyhex_custom(hrows, hcols)
        n_hex = hrows * hcols
        assert nx.is_connected(G)
        print(f"  ✓ Heavy-Hex {hrows}×{hcols} ({n_hex} hex cells): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    _sanity()
    print("✓ Sanity tests passed!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "test":
            test_all_layouts()
        elif mode == "terminal":
            print_heavyhex_terminal(2, 2)
            print()
            print_heavyhex_terminal(2, 3)
        else:
            print("Usage: python quantum_layouts.py [test|terminal]")
    else:
        test_all_layouts()

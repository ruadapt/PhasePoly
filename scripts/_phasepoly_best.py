"""Helpers for the per-tag `phasepoly_best_<tag>/` summary.

After each circuit finishes, run_experiment.py calls update_phasepoly_best() to:
  1. copy {circuit}(best).qasm + .txt into <tag>/phasepoly_best_<tag>/
  2. upsert one row in <tag>/phasepoly_best_<tag>/summary.csv

The CSV upsert uses fcntl.flock so multiple tmux sessions sharing a tag
can safely write concurrently. Each row is keyed by circuit_name; re-running
the same circuit replaces its row instead of appending.
"""
import csv
import fcntl
import json
import shutil
from pathlib import Path
from typing import Optional

CSV_HEADER = [
    "circuit_name",
    "model",
    "is_timeout",
    "total_gate_count",
    "weighted_cx",
    "rz_gate_count",
    "t_gate_count",
    "weighted_depth",
    "synthesis_time",
    "total_time",
]


def _load_progress(circuit_dir: Path) -> Optional[dict]:
    """Prefer result.json (final); fall back to _progress.json (mid-flight)."""
    for fname in ("result.json", "_progress.json"):
        p = circuit_dir / fname
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def build_row(circuit_dir: Path, is_timeout: bool, model: str) -> dict:
    """Build the summary row dict for one circuit's results."""
    circuit_name = circuit_dir.name
    row = {
        "circuit_name": circuit_name,
        "model": model,
        "is_timeout": "true" if is_timeout else "false",
        "total_gate_count": "",
        "weighted_cx": "",
        "rz_gate_count": "",
        "t_gate_count": "",
        "weighted_depth": "",
        "synthesis_time": "",
        "total_time": "",
    }
    progress = _load_progress(circuit_dir)
    if progress is None:
        return row

    # Time columns sum across ALL recorded rounds, including failed rounds when
    # phasepoly.py was able to return a partial result.
    total_synth = 0.0
    total_wall = 0.0
    for round_data in progress.get("rounds", {}).values():
        try:
            total_synth += float(round_data.get("synthesis_time", 0) or 0)
        except (TypeError, ValueError):
            pass
        try:
            total_wall += float(round_data.get("total_time", 0) or 0)
        except (TypeError, ValueError):
            pass
    row["synthesis_time"] = f"{total_synth:.6f}"
    row["total_time"] = f"{total_wall:.6f}"

    # Metrics come from the last successful round's circuit_info. If no round
    # succeeded, the best artifact is the original input, so use the first
    # recorded input_circuit_info when available.
    last = progress.get("last_successful_round", 0)
    if last and str(last) in progress.get("rounds", {}):
        info = progress["rounds"][str(last)].get("circuit_info") or {}
    else:
        info = {}
        for round_data in progress.get("rounds", {}).values():
            info = round_data.get("input_circuit_info") or {}
            if info:
                break
    if info:
        row["total_gate_count"] = info.get("gates(always weighted)", "")
        row["weighted_cx"] = info.get("weighted_cx", "")
        row["rz_gate_count"] = info.get("rz_gate", "")
        row["t_gate_count"] = info.get("t_gate", "")
        row["weighted_depth"] = info.get("weighted_depth", "")
    return row


def _upsert_csv_row(csv_path: Path, row: dict) -> None:
    """Multi-process-safe upsert: replace the row keyed by circuit_name, or append."""
    csv_path.touch(exist_ok=True)
    with open(csv_path, "r+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            content = f.read()
            existing = []
            if content.strip():
                existing = list(csv.DictReader(content.splitlines()))
            replaced = False
            for i, r in enumerate(existing):
                if r.get("circuit_name") == row["circuit_name"]:
                    existing[i] = row
                    replaced = True
                    break
            if not replaced:
                existing.append(row)
            f.seek(0)
            f.truncate()
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            for r in existing:
                writer.writerow({k: r.get(k, "") for k in CSV_HEADER})
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


PHASEPOLY_BEST_DIRNAME = "phasepoly_best"


def phasepoly_best_dirname(tag_dir: Path) -> str:
    return f"{PHASEPOLY_BEST_DIRNAME}_{tag_dir.name}"


def update_phasepoly_best(tag_dir: Path, circuit_name: str, is_timeout: bool,
                          model: str = "phasepoly") -> None:
    """Per-circuit: copy (best) artifacts + upsert one CSV row."""
    phasepoly_best_dir = tag_dir / phasepoly_best_dirname(tag_dir)
    phasepoly_best_dir.mkdir(parents=True, exist_ok=True)
    circuit_dir = tag_dir / circuit_name
    for ext in ("qasm", "txt"):
        src = circuit_dir / f"{circuit_name}(best).{ext}"
        if src.exists():
            shutil.copyfile(src, phasepoly_best_dir / f"{circuit_name}(best).{ext}")
    row = build_row(circuit_dir, is_timeout, model)
    _upsert_csv_row(phasepoly_best_dir / "summary.csv", row)


def rebuild_phasepoly_best(tag_dir: Path, model: str = "phasepoly") -> int:
    """Batch: rebuild phasepoly_best_<tag>/ from scratch by walking every circuit subdir.
    Used by aggregate_results.py. Returns number of circuits processed.
    Cannot detect timeout retroactively — uses presence of result.json
    + last_successful_round to mark is_timeout."""
    if not tag_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {tag_dir}")
    phasepoly_best_dir = tag_dir / phasepoly_best_dirname(tag_dir)
    csv_path = phasepoly_best_dir / "summary.csv"
    phasepoly_best_dir.mkdir(exist_ok=True)
    # Reset CSV so rebuild reflects only currently-present circuits.
    if csv_path.exists():
        csv_path.unlink()
    count = 0
    for child in sorted(tag_dir.iterdir()):
        if not child.is_dir() or child.name.startswith(PHASEPOLY_BEST_DIRNAME):
            continue
        # Heuristic: circuits whose worker was killed have _progress.json
        # but no result.json. Treat as is_timeout=true for the rebuild.
        is_timeout = (child / "_progress.json").exists() and not (child / "result.json").exists()
        update_phasepoly_best(tag_dir, child.name, is_timeout=is_timeout, model=model)
        count += 1
    return count

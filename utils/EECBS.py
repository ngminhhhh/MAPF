from __future__ import annotations

import csv
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DELTA_TO_ACTION = {
    (0, 0): 0,
    (-1, 0): 1,
    (1, 0): 2,
    (0, -1): 3,
    (0, 1): 4,
}
COORD_PATTERN = re.compile(r"\((\d+),(\d+)\)")


@dataclass
class EECBSResult:
    positions: np.ndarray
    expert_actions: np.ndarray
    raw_paths: list[np.ndarray]
    makespan: int
    solution_cost: float | None


def _windows_to_wsl_path(path: str | Path) -> str:
    path_str = str(Path(path).resolve())
    drive, _, tail = path_str.partition(":")
    if not tail:
        return path_str.replace("\\", "/")
    return f"/mnt/{drive.lower()}{tail.replace(chr(92), '/')}"


def _write_map(grid: np.ndarray, map_path: Path) -> None:
    height, width = grid.shape
    lines = [
        "type octile",
        f"height {height}",
        f"width {width}",
        "map",
    ]

    for row in grid:
        lines.append("".join("." if cell == 0 else "@" for cell in row))

    map_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scen(
    map_name: str,
    grid: np.ndarray,
    agent_pos: np.ndarray,
    agent_goal: np.ndarray,
    scen_path: Path,
) -> None:
    height, width = grid.shape
    lines = ["version 1"]

    for (rs, cs), (rg, cg) in zip(agent_pos, agent_goal):
        lines.append(f"0\t{map_name}\t{width}\t{height}\t{cs}\t{rs}\t{cg}\t{rg}\t0")

    scen_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_eecbs(
    solver_exec: Path,
    map_path: Path,
    scen_path: Path,
    stats_path: Path,
    paths_path: Path,
    n_agents: int,
    cutoff_time: float,
    suboptimality: float,
    screen: int,
) -> subprocess.CompletedProcess[str]:
    command = " ".join(
        [
            shlex.quote(_windows_to_wsl_path(solver_exec)),
            "-m",
            shlex.quote(_windows_to_wsl_path(map_path)),
            "-a",
            shlex.quote(_windows_to_wsl_path(scen_path)),
            "-o",
            shlex.quote(_windows_to_wsl_path(stats_path)),
            "--outputPaths",
            shlex.quote(_windows_to_wsl_path(paths_path)),
            "-k",
            str(n_agents),
            "-t",
            str(cutoff_time),
            "-s",
            str(screen),
            "--suboptimality",
            str(suboptimality),
        ]
    )

    return subprocess.run(
        ["wsl", "bash", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
    )


def _parse_paths(paths_path: Path, n_agents: int) -> list[np.ndarray]:
    raw_paths = []

    for line in paths_path.read_text(encoding="utf-8").splitlines():
        coords = [(int(r), int(c)) for r, c in COORD_PATTERN.findall(line)]
        if coords:
            raw_paths.append(np.asarray(coords, dtype=np.int64))

    if len(raw_paths) != n_agents:
        raise RuntimeError(f"Expected {n_agents} paths, got {len(raw_paths)}")

    return raw_paths


def _build_positions(raw_paths: list[np.ndarray]) -> np.ndarray:
    max_len = max(len(path) for path in raw_paths)
    positions = np.empty((max_len, len(raw_paths), 2), dtype=np.int64)

    for agent_idx, path in enumerate(raw_paths):
        positions[: len(path), agent_idx] = path
        positions[len(path) :, agent_idx] = path[-1]

    return positions


def _build_actions(positions: np.ndarray) -> np.ndarray:
    deltas = positions[1:] - positions[:-1]
    actions = np.empty(deltas.shape[:2], dtype=np.int64)

    for t in range(deltas.shape[0]):
        for agent_idx in range(deltas.shape[1]):
            delta = tuple(deltas[t, agent_idx].tolist())
            if delta not in DELTA_TO_ACTION:
                raise RuntimeError(f"Unexpected move delta {delta}")
            actions[t, agent_idx] = DELTA_TO_ACTION[delta]

    return actions


def _read_solution_cost(stats_path: Path) -> float | None:
    if not stats_path.exists():
        return None

    with stats_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    value = rows[-1].get("solution cost")
    return None if value in (None, "") else float(value)


def eecbs(
    grid: np.ndarray,
    agent_pos: np.ndarray,
    agent_goal: np.ndarray,
    *,
    instance_id: int | str | None = None,
    solver_dir: str | Path = "EECBS",
    cutoff_time: float = 60.0,
    suboptimality: float = 1.2,
    screen: int = 0,
    keep_files: bool = False,
) -> EECBSResult:
    solver_exec = Path(solver_dir).resolve() / "eecbs"
    suffix = f"_{instance_id}" if instance_id is not None else ""
    artifacts_dir = Path(tempfile.mkdtemp(prefix=f"run{suffix}_", dir=solver_exec.parent))

    map_path = artifacts_dir / f"instance{suffix}.map"
    scen_path = artifacts_dir / f"instance{suffix}.scen"
    stats_path = artifacts_dir / f"instance{suffix}.csv"
    paths_path = artifacts_dir / f"instance{suffix}_paths.txt"

    _write_map(grid, map_path)
    _write_scen(map_path.name, grid, agent_pos, agent_goal, scen_path)

    result = _run_eecbs(
        solver_exec=solver_exec,
        map_path=map_path,
        scen_path=scen_path,
        stats_path=stats_path,
        paths_path=paths_path,
        n_agents=agent_pos.shape[0],
        cutoff_time=cutoff_time,
        suboptimality=suboptimality,
        screen=screen,
    )

    if result.returncode != 0:
        raise RuntimeError(f"EECBS failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    if not paths_path.exists() or paths_path.stat().st_size == 0:
        raise RuntimeError(f"EECBS did not produce paths.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    raw_paths = _parse_paths(paths_path, agent_pos.shape[0])
    positions = _build_positions(raw_paths)
    expert_actions = _build_actions(positions)
    solution_cost = _read_solution_cost(stats_path)

    if not keep_files:
        shutil.rmtree(artifacts_dir)

    return EECBSResult(
        positions=positions,
        expert_actions=expert_actions,
        raw_paths=raw_paths,
        makespan=positions.shape[0] - 1,
        solution_cost=solution_cost,
    )

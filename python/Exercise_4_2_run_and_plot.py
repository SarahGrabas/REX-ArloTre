"""Kør Exercise 4 på robotten og gem kun JSON; kræver ikke Matplotlib.

Fra robotprojektets python-mappe:
    python3 Exercise_4_1.py && python3 Exercise_4_2_run_and_plot.py --run

--run udfører den normale fysiske robotkørsel. Efter afslutning gemmes grid,
landmarks og ruter i plot_grids/Exercise_4_2_grid_plot_(N).json.
Overfør JSON til computeren og brug Exercise_4_2_plot_from_json.py til PNG.
Koordinatfilen læses fra samme mappe som Exercise_4_2.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import runpy
import sys
from typing import Any, Literal, Mapping, TypedDict, overload, Optional, Union

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PREFIX = 'Exercise_4_2_grid_plot_'


class LandmarkData(TypedDict):
    id: int
    center: list[float]
    radius: float


class PlotData(TypedDict):
    grid_matrix: list[list[bool]]
    x_limits: list[float]
    y_limits: list[float]
    grid_cell_size: float
    robot_radius: float
    coordinates: list[LandmarkData]
    path: Optional[list[list[float]]]
    simpler_path: Optional[list[list[float]]]
    execute_path: Optional[list[list[float]]]
    start: Optional[list[float]]
    goal: Optional[list[float]]


def capture_result(namespace: Mapping[str, Any]) -> PlotData:
    """Gem selve matrixen: ingen genberegning af grid eller ny RRT-kørsel."""
    grid = namespace['map']

    @overload
    def points(name: Literal['START_POINT', 'GOAL']) -> Optional[list[float]]: ...

    @overload
    def points(name: Literal['path', 'simpler_path', 'execute_path']) -> Optional[list[list[float]]]: ...

    def points(name: str) -> Optional[Union[list[float], list[list[float]]]]:
        value = namespace.get(name)
        return None if value is None else np.asarray(value, dtype=float).tolist()

    return {
        'grid_matrix': np.asarray(grid.grid_matrix, dtype=bool).tolist(),
        'x_limits': list(grid.x_limits),
        'y_limits': list(grid.y_limits),
        'grid_cell_size': float(grid.grid_cell_size),
        'robot_radius': float(namespace['arlo_radius']),
        'coordinates': [
            {'id': int(marker_id), 'center': np.asarray(center, dtype=float).tolist(),
             'radius': float(radius)}
            for marker_id, center, radius in namespace['landmarks_list']
        ],
        'path': points('path'),
        'simpler_path': points('simpler_path'),
        'execute_path': points('execute_path'),
        'start': points('START_POINT'),
        'goal': points('GOAL'),
    }


def run_exercise() -> PlotData:
    """Kører den eksisterende hovedfil uændret, inklusive motorstyringen."""
    previous_cwd = Path.cwd()
    previous_path = sys.path[:]
    try:
        os.chdir(SCRIPT_DIR)
        sys.path.insert(0, str(SCRIPT_DIR))
        namespace = runpy.run_path(str(SCRIPT_DIR / 'Exercise_4_2.py'), run_name='__main__')
        return capture_result(namespace)
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


def save_data(data: PlotData, output_dir: Union[str, os.PathLike[str]]) -> Path:
    """Gem JSON med næste ledige nummer uden at importere Matplotlib."""
    # Valider serialisering inden oprettelse af filen.
    serialized = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r'Exercise_4_2_grid_plot_\((\d+)\)\.(png|json)$')
    numbers = [int(match.group(1)) for path in output_dir.iterdir()
               if (match := pattern.fullmatch(path.name))]
    number = max(numbers, default=0) + 1
    while True:
        json_path = output_dir / f'{PREFIX}({number}).json'
        if json_path.with_suffix('.png').exists():
            number += 1
            continue
        try:
            stream = json_path.open('x', encoding='utf-8')
            break
        except FileExistsError:
            number += 1
    with stream:
        stream.write(serialized)
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='store_true', required=True,
                        help='Kør Exercise_4_2.py inklusive robotkørsel, og gem kun JSON')
    parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR / 'plot_grids')
    args = parser.parse_args()
    print('Kører Exercise_4_2.py inklusive robotkørsel. JSON gemmes efter afslutning.', flush=True)
    data = run_exercise()
    json_path = save_data(data, args.output_dir.resolve())
    print(f'Data gemt: {json_path}')


if __name__ == '__main__':
    main()

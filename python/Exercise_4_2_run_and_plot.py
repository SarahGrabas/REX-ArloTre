"""Kør Exercise 4 og gem det faktiske grid og ruter uden at ændre robotfilerne.

På robotten (udfører også den normale robotkørsel):
    python3 Exercise_4_2_run_and_plot.py --run

Genplot gemte data uden robotforbindelse:
    python3 Exercise_4_2_run_and_plot.py --input "plot_grids/Exercise_4_2_grid_plot_(1).json"

PNG og JSON gemmes ved siden af hinanden i python/plot_grids. --show åbner
desuden figuren. --output-dir kan bruges til at vælge en anden mappe.
Koordinatfilen til --run læses fra samme mappe som Exercise_4_2.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import runpy
import sys
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypedDict, overload, Optional, Union

if TYPE_CHECKING:
    from matplotlib.figure import Figure

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


def make_plot(data: PlotData) -> Figure:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Patch, Rectangle

    grid = np.asarray(data['grid_matrix'], dtype=bool)
    low = np.array([data['x_limits'][0], data['y_limits'][0]], dtype=float)
    high = np.array([data['x_limits'][1], data['y_limits'][1]], dtype=float)
    cell_size = float(data['grid_cell_size'])
    if grid.ndim != 2 or not grid.size or cell_size <= 0 or np.any(high <= low):
        raise ValueError('Ugyldigt grid eller ugyldige kortgrænser i input')
    # Brug faktiske cellekanter, også hvis grid er mindre end kortgrænsen.
    xs = np.minimum(low[0] + np.arange(grid.shape[0] + 1) * cell_size, high[0])
    ys = np.minimum(low[1] + np.arange(grid.shape[1] + 1) * cell_size, high[1])
    coordinates = data['coordinates']
    fig, axes = plt.subplots(1, 2, figsize=(12, 9), gridspec_kw={'width_ratios': [1, 1.5]})
    fig.suptitle('Exercise 4.2 – grid og ruter fra samme kørsel', fontsize=15)
    route_styles = [
        ('path', '.--', '#287bb5', 'RRT-rute', 1.2),
        ('simpler_path', '--', '#15956a', 'Simpler path', 2.8),
        ('execute_path', 'o-', '#ce3047', 'execute_path (kommandoer)', 1.5),
    ]
    for ax in axes:
        ax.pcolormesh(xs, ys, grid.T, cmap=ListedColormap(['#ffffff', '#c7cdd5']),
                      vmin=0, vmax=1, edgecolors='#e3e6eb', linewidth=0.35)
        ax.add_patch(Rectangle(low, *(high-low), fill=False, linestyle='--',
                               edgecolor='black', label='Angivet kortgrænse'))
        for i, item in enumerate(coordinates):
            center = item['center'][:2]
            radius = item['radius']
            ax.add_patch(Circle(center, radius, color='#efaa45', alpha=0.65,
                                label='Landmarkets modelcirkel' if i == 0 else None))
            # Bufferen er allerede med i den gemte gridmatrix. Her vises kun
            # fysisk modelradius + robotradius, så vi ikke gætter på en buffer.
            ax.add_patch(Circle(center, radius + data['robot_radius'], fill=False,
                                linestyle=':', edgecolor='#91652c',
                                label='Landmarkradius + robotradius' if i == 0 else None))
            ax.text(*center, str(item['id']), ha='center', va='center',
                    fontweight='bold', clip_on=True)
        for key, style, color, label, width in route_styles:
            route = data.get(key)
            if route is not None and len(route):
                points = np.asarray(route, dtype=float)
                ax.plot(*points[:, :2].T, style, color=color, linewidth=width,
                        markersize=4, label=label)
        for key, label, marker, color in [('start', 'Start', 'o', 'black'),
                                           ('goal', 'Mål', '*', '#15956a')]:
            if data.get(key) is not None:
                ax.scatter(*data[key][:2], marker=marker, color=color, s=65,
                           zorder=6, label=label)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meter)')
        ax.set_ylabel('Y (meter)')

    # Oversigten inkluderer også landmarks, som ligger uden for kortet.
    bounds_low, bounds_high = low.copy(), high.copy()
    for item in coordinates:
        c, r = np.asarray(item['center'][:2]), item['radius']
        bounds_low = np.minimum(bounds_low, c-r)
        bounds_high = np.maximum(bounds_high, c+r)
    axes[0].set(xlim=(bounds_low[0]-.15, bounds_high[0]+.15),
                ylim=(bounds_low[1]-.15, bounds_high[1]+.15),
                title=f'Hele grid: {grid.shape[0]} × {grid.shape[1]} celler')
    # Zoom omkring landmarks, ikke en hardkodet placering fra en gammel rute.
    if coordinates:
        centers = np.array([item['center'][:2] for item in coordinates])
        radii = np.array([item['radius'] + data['robot_radius'] for item in coordinates])
        zoom_low = np.min(centers-radii[:, None], axis=0)-.15
        zoom_high = np.max(centers+radii[:, None], axis=0)+.15
    else:
        zoom_low, zoom_high = low-.15, high+.15
    axes[1].set(xlim=(zoom_low[0], zoom_high[0]), ylim=(zoom_low[1], zoom_high[1]),
                title='Zoom omkring landmarks')
    if data.get('path') is None:
        axes[0].text(.5, .97, 'Ingen rute fundet', transform=axes[0].transAxes,
                     ha='center', va='top', color='#ce3047',
                     bbox={'facecolor': 'white', 'alpha': .9})
    handles, labels = axes[0].get_legend_handles_labels()
    handles.insert(0, Patch(facecolor='#c7cdd5'))
    labels.insert(0, 'Optaget celle (inkl. gridbuffer)')
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .04), ncol=3, fontsize=9)
    fig.text(.5, .015,
             f'Celler: {cell_size*100:g} × {cell_size*100:g} cm. '
             'execute_path er kommanderede punkter, ikke en målt robotbane.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .19, 1, .95))
    return fig


def save_plot(data: PlotData, output_dir: Union[str, os.PathLike[str]]) -> tuple[Figure, Path, Path]:
    """Næste nummer er største eksisterende nummer + 1, overskriv aldrig."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r'Exercise_4_2_grid_plot_\((\d+)\)\.(png|json)$')
    numbers = [int(match.group(1)) for path in output_dir.iterdir()
               if (match := pattern.fullmatch(path.name))]
    number = max(numbers, default=0) + 1
    fig = make_plot(data)
    while True:
        png_path = output_dir / f'{PREFIX}({number}).png'
        json_path = png_path.with_suffix('.json')
        if json_path.exists():
            number += 1
            continue
        try:
            # Eksklusiv oprettelse beskytter også mod samtidige kørsler.
            stream = png_path.open('xb')
            break
        except FileExistsError:
            number += 1
    with stream:
        fig.savefig(stream, format='png', dpi=180)
    with json_path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return fig, png_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', action='store_true',
                        help='Kør Exercise_4_2.py, inklusive fysisk robotkørsel, og gem plot')
    source.add_argument('--input', type=Path, help='Genplot en gemt JSON uden robotforbindelse')
    parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR / 'plot_grids')
    parser.add_argument('--show', action='store_true', help='Åbn også plotvinduet')
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if args.run:
        print('Kører Exercise_4_2.py inklusive robotkørsel. Plot gemmes efter afslutning.', flush=True)
        data = run_exercise()
    else:
        with args.input.open(encoding='utf-8') as stream:
            data = json.load(stream)
    fig, png_path, json_path = save_plot(data, args.output_dir.resolve())
    print(f'Plot gemt: {png_path}')
    print(f'Data gemt: {json_path}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()

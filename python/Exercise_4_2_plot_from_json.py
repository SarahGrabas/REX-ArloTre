"""Lav PNG-plots på computeren fra JSON-filer overført fra robotten.

En fil:
    python3 Exercise_4_2_plot_from_json.py --input "plot_grids/Exercise_4_2_grid_plot_(1).json"
Alle nummererede JSON-filer i en mappe:
    python3 Exercise_4_2_plot_from_json.py --input plot_grids

PNG gemmes ved siden af JSON med samme navn og nummer. JSON ændres ikke.
Brug --show til også at åbne plotvinduer, --output-dir til en anden mappe,
eller --overwrite til at erstatte eksisterende PNG-filer.
Denne fil starter ikke robotten og genberegner ikke ruten eller gridet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from Exercise_4_2_run_and_plot import PlotData


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


def plot_json(input_path: Path, output_dir: Path | None = None,
              overwrite: bool = False) -> tuple[Figure, Path]:
    """Læs et snapshot og gem PNG med samme nummer; behold JSON uændret."""
    import matplotlib.pyplot as plt

    png_path = (input_path.parent if output_dir is None else output_dir) / (input_path.stem + '.png')
    if png_path.exists() and not overwrite:
        raise FileExistsError(png_path)
    with input_path.open(encoding='utf-8') as stream:
        data: PlotData = json.load(stream)
    fig = make_plot(data)
    try:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        with png_path.open('wb' if overwrite else 'xb') as stream:
            fig.savefig(stream, format='png', dpi=180)
    except Exception:
        plt.close(fig)
        raise
    return fig, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, nargs='+', required=True,
                        help='En eller flere JSON-filer eller mapper med nummererede JSON-filer')
    parser.add_argument('--output-dir', type=Path, help='Standard: samme mappe som hver JSON-fil')
    parser.add_argument('--show', action='store_true', help='Åbn også plotvinduer')
    parser.add_argument('--overwrite', action='store_true', help='Erstat eksisterende PNG-filer')
    args = parser.parse_args()

    files: list[Path] = []
    for source in args.input:
        if source.is_dir():
            files.extend(sorted(source.glob('Exercise_4_2_grid_plot_(*).json')))
        elif source.is_file() and source.suffix.lower() == '.json':
            files.append(source)
        else:
            parser.error(f'Ikke en JSON-fil eller mappe: {source}')
    if not files:
        parser.error('Ingen nummererede JSON-filer fundet')

    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for input_path in dict.fromkeys(path.resolve() for path in files):
        try:
            fig, png_path = plot_json(input_path, args.output_dir, args.overwrite)
        except FileExistsError as error:
            print(f'Springer over eksisterende PNG: {error}. Brug --overwrite for at erstatte.')
            continue
        print(f'Plot gemt: {png_path}')
        if not args.show:
            plt.close(fig)
    if args.show:
        plt.show()
        plt.close('all')


if __name__ == '__main__':
    main()

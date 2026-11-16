#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   plot_objective.py
@Time    :   2026/01/15 17:39:00
@Author  :   George Trenins
@Desc    :   Plot the objective function and the embedding approximation along the trajectory followed by the optimizer
'''

from __future__ import print_function, division, absolute_import
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from glefit.config.config_handler import ConfigHandler
from glefit.embedding import BaseEmbedder, EMBEDDER_MAP, MultiEmbedder
from glefit.merit import BaseProperty, PROPERTY_MAP


def parse_trajectory_file(filepath: Path) -> tuple[list[int], list[np.ndarray]]:
    """Parse trajectory file to extract step numbers and parameter arrays.
    
    Returns:
        Tuple of (step_numbers, parameter_arrays)

    Raises:
        ValueError: if the file has no step, as when the fit stopped
            before it wrote step 0, or the file is not a trajectory.
    """
    steps = []
    params = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '|' in line and 'Step' in line:
            # Extract step number
            step_str = line.split('Step')[1].split(',')[0].strip()
            step_num = int(step_str)
            steps.append(step_num)
            
            # Collect array lines (between header and blank line)
            i += 1
            array_str = ""
            while i < len(lines) and lines[i].strip():
                array_str += lines[i].strip()
                i += 1
            
            # Parse array - remove brackets and parse
            array_str = array_str.strip('[]')
            param_array = np.fromstring(array_str, sep=',')
            params.append(param_array)
        i += 1

    if not steps:
        raise ValueError(f"no steps in the trajectory file {filepath}")
    return steps, params


def parse_log_file(
        filepath: Path
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Read the step numbers, distances and max|gᵢ| from the log of a fit.

    The log is the one that NewtonRaphson.log writes, where a line
    `<Class>: step hh:mm:ss distance max|gᵢ|`, with 8 significant digits,
    follows each header `Step Time Value Gmax`.

    Returns:
        Tuple of (step_numbers, distances, max_gradients)

    Raises:
        ValueError: if the file has no step.
    """
    steps, distance, gmax = [], [], []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for header, line in zip(lines, lines[1:]):
        if header.split() == ["Step", "Time", "Value", "Gmax"]:
            _, step, _, value, grad = line.split()
            steps.append(int(step))
            distance.append(float(value))
            gmax.append(float(grad))
    if not steps:
        raise ValueError(f"no steps in the log file {filepath}")
    return steps, np.array(distance), np.array(gmax)


def setup_from_config(config_path: Path) -> tuple[BaseEmbedder, BaseProperty]:
    """Load config and create embedder and merit function."""
    handler = ConfigHandler(config_path)
    handler.validate()
    
    # Create embedder
    embedder_config = handler.get_embedder_config()
    EmbedderClass = EMBEDDER_MAP[embedder_config["type"]]
    embedder = EmbedderClass.from_dict(embedder_config["parameters"])

    # Turn off default constraints in the embedder
    if hasattr(embedder, '_embs'):
        for emb in embedder._embs:
            emb._mappers = None
    else:
        embedder._mappers = None
    
    # Load data and create merit function
    datasets = handler.load_data()
    merit_config = handler.get_merit_function_config()
    PropertyClass = PROPERTY_MAP[merit_config["type"]]
    merit = PropertyClass.from_dict(
        merit_config["parameters"], 
        datasets, 
        embedder
    )
    
    return embedder, merit


def compute_contributions(
    embedder: BaseEmbedder, 
    merit: BaseProperty,
    params: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """Compute property contributions for given parameters.
    
    Returns:
        Tuple of (contributions_list, total) where:
        - contributions_list: List of component contributions (empty if not MultiEmbedder)
        - total: Sum of all contributions
    """
    # Set embedder parameters
    embedder.conventional_params = params
    
    # Check if MultiEmbedder
    if isinstance(embedder, MultiEmbedder):
        contributions = []
        for component_emb in embedder._embs:
            merit.emb = component_emb
            contributions.append(merit.function())
        merit.emb = embedder
        total = merit.function()
        return contributions, total
    else:
        # Single embedder - just one contribution
        total = merit.function()
        return [total], total

def plot_trajectory(
    merit: BaseProperty,
    embedder: BaseEmbedder,
    steps: list[int],
    params_list: list[np.ndarray],
    log: Optional[tuple[list[int], np.ndarray, np.ndarray]] = None
) -> tuple[plt.Figure, Slider]:
    """Create interactive plot with slider.

    With `log`, the step numbers, distances and max|gᵢ| that
    parse_log_file returns, a second panel shows the distance and max|gᵢ|
    against step, with a line at the selected step.

    Returns:
        The figure and the slider, which must stay referenced for the
        slider to respond.

    Raises:
        ValueError: if the log and the trajectory have different steps.
    """
    if log is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        if list(log[0]) != list(steps):
            raise ValueError(
                "the steps of the log differ from those of the trajectory")
        fig, (ax, ax_log) = plt.subplots(
            2, 1, figsize=(10, 9), height_ratios=[2, 1])
    plt.subplots_adjust(bottom=0.25, right=0.85)
    
    ax.set_xlabel('Grid')
    ax.set_ylabel('Property value')
    contributions, total = compute_contributions(embedder, merit, params_list[0])
    is_multi = len(contributions) > 1
    
    # Compute initial merit value
    merit_value = merit.distance
    
    # Set title with merit value
    ax.set_title(f'Step {steps[0]} / {steps[-1]} | Merit = {merit_value:.6e}')
    
    # Plot target
    ax.plot(
        merit.grid, merit.target, 
        'r-', label='Target', linewidth=1
    )
    
    # Set fixed axis ranges
    y_min, y_max = min(0, merit.target.min()), merit.target.max()
    y_range = y_max - y_min
    ax.set_xlim(merit.grid.min(), merit.grid.max())
    ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
    
    # Initialize plot lines
    contribution_lines = []
    if is_multi and len(contributions) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, len(contributions)))
        for i, contrib in enumerate(contributions):
            line, = ax.plot(
                merit.grid, contrib, '--', 
                color=colors[i],
                label=f'Component {i+1}',
                linewidth=1
            )
            contribution_lines.append(line)
    
    # Plot total
    total_line, = ax.plot(
        merit.grid, total, 
        'k-', label='Total', linewidth=1
    )
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)

    # Plot the distance and max|gᵢ| against step, and mark the step
    if log is not None:
        log_steps, distance, gmax = log
        ax_log.semilogy(log_steps, distance, 'k.-', label='Distance',
                        linewidth=1)
        ax_log.semilogy(log_steps, gmax, 'b.-', label=r'max $|g_i|$',
                        linewidth=1)
        marker = ax_log.axvline(steps[0], color='r', label='Selected step',
                                linewidth=1)
        ax_log.set_xlabel('Step')
        # inside, below where both curves start
        ax_log.legend(loc='lower left')
        ax_log.grid(True, alpha=0.3)

    # Create slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider, 'Step', 
        0, len(steps)-1, 
        valinit=0, valstep=1
    )
    
    def update(val):
        step_idx = int(slider.val)
        
        # Compute contributions for this step
        contributions_step, total_step = compute_contributions(
            embedder, merit, params_list[step_idx]
        )
        
        # Update contribution lines
        if is_multi and len(contributions_step) > 0:
            for i, line in enumerate(contribution_lines):
                line.set_ydata(contributions_step[i])
        
        # Update total line
        total_line.set_ydata(total_step)
        
        # Compute merit value
        merit_value = merit.distance
        
        # Update title
        ax.set_title(f'Step {steps[step_idx]} / {steps[-1]} | Merit = {merit_value:.6e}')

        if log is not None:
            marker.set_xdata([steps[step_idx]] * 2)

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    return fig, slider


def main():
    parser = argparse.ArgumentParser(
        description="Visualize optimization trajectory and property fit"
    )
    parser.add_argument('config', type=str, help="Path to config file")
    parser.add_argument(
        '--traj', '-t',
        type=str, 
        default='traj.out',
        help="Path to trajectory file (default: traj.out)"
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=None,
        help="Path to the log file of the fit, to plot the distance and "
             "max|g_i| against step (default: none)"
    )
    args = parser.parse_args()
    
    # Parse trajectory
    print("Parsing trajectory file...")
    traj_path = Path(args.traj)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    
    steps, params_list = parse_trajectory_file(traj_path)
    print(f"Found {len(steps)} optimization steps")

    log = None
    if args.log is not None:
        print("Parsing log file...")
        log = parse_log_file(Path(args.log))

    # Load configuration
    print("Loading configuration...")
    config_path = Path(args.config)
    embedder, merit = setup_from_config(config_path)
    
    # Create interactive plot
    print("Creating interactive plot...")
    plot_trajectory(merit, embedder, steps, params_list, log=log)


if __name__ == "__main__":
    main()
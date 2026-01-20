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
    
    return steps, params


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
        emb._mappers = None
    
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
    params_list: list[np.ndarray]
):
    """Create interactive plot with slider."""
    fig, ax = plt.subplots(figsize=(10, 6))
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
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()


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
    args = parser.parse_args()
    
    # Parse trajectory
    print("Parsing trajectory file...")
    traj_path = Path(args.traj)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    
    steps, params_list = parse_trajectory_file(traj_path)
    print(f"Found {len(steps)} optimization steps")
    
    # Load configuration
    print("Loading configuration...")
    config_path = Path(args.config)
    embedder, merit = setup_from_config(config_path)
    
    # Create interactive plot
    print("Creating interactive plot...")
    plot_trajectory(merit, embedder, steps, params_list)


if __name__ == "__main__":
    main()
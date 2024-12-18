import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import os

from utils.space import Space
from utils.envs import PassiveHapticsEnv


def plot_physical_space(ax, physical_space, label_prefix=''):
    """
    Draws the physical space boundaries and obstacles.
    
    Parameters:
    - ax: Matplotlib axes object.
    - physical_space: Space object containing boundary and obstacle information.
    - label_prefix: Prefix for labels to distinguish physical and virtual spaces.
    """
    # Draw physical boundary
    border = physical_space.border
    mpl_polygon = MplPolygon(border, closed=True, fill=None, edgecolor='black', linewidth=2,
                             label=f'{label_prefix}Physical Border')
    ax.add_patch(mpl_polygon)

    # Draw physical obstacles
    for idx, obstacle in enumerate(physical_space.obstacle_list):
        mpl_obstacle = MplPolygon(obstacle, closed=True, facecolor='gray', edgecolor='red', alpha=0.7)
        if idx == 0:
            mpl_obstacle.set_label(f'{label_prefix}Obstacle')
        ax.add_patch(mpl_obstacle)


def plot_path(ax, path_x, path_y, label='Path', cmap_name='Blues'):
    """
    Draws a path with a color gradient and marks the start and end points.
    
    Parameters:
    - ax: Matplotlib axes object.
    - path_x: List of x-coordinates.
    - path_y: List of y-coordinates.
    - label: Label for the path.
    - cmap_name: Name of the colormap to use.
    
    Returns:
    - Line2D object for the legend.
    """
    if len(path_x) == 0 or len(path_y) == 0:
        print(f"Path '{label}' is empty, cannot plot.")
        return None

    # Create segments for the path
    points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Apply reversed colormap for deeper colors towards the end
    cmap = plt.get_cmap(cmap_name).reversed()
    norm = plt.Normalize(0, len(segments))
    colors = [cmap(norm(i)) for i in range(len(segments))]

    # Create LineCollection
    lc = LineCollection(segments, colors=colors, linewidth=2)
    ax.add_collection(lc)

    # Mark start and end points
    ax.plot(path_x[0], path_y[0], marker='s', color='green', markersize=8, label=f'{label} Start')
    ax.plot(path_x[-1], path_y[-1], marker='*', color='gold', markersize=12, label=f'{label} End')

    # Create a representative line for the legend
    path_legend_color = cmap(norm(0))
    path_line = Line2D([0], [0], color=path_legend_color, lw=2, label=label)
    return path_line


def plot_fig(physical_space, s2c_x_l, s2c_y_l, s2o_x_l, s2o_y_l, srl_x_l, srl_y_l, s2mt_x_l, s2mt_y_l, arc_x_l, arc_y_l, apf_x_l, apf_y_l, none_x_l, none_y_l, path_index):
    """
    Plots a single path with multiple methods on a new figure.
    
    Parameters:
    - physical_space: Space object.
    - s2c_x_l, s2c_y_l: S2C method path coordinates.
    - s2o_x_l, s2o_y_l: S2O method path coordinates.
    - srl_x_l, srl_y_l: SRL method path coordinates.
    - path_index: Index of the current path for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_physical_space(ax, physical_space)

    # Plot each method's path and collect legend handles
    path_lines = []
    path_lines.append(plot_path(ax, s2c_x_l, s2c_y_l, label='S2C', cmap_name='Blues'))
    path_lines.append(plot_path(ax, s2o_x_l, s2o_y_l, label='S2O', cmap_name='Greens'))
    path_lines.append(plot_path(ax, srl_x_l, srl_y_l, label='SRL', cmap_name='Reds'))
    path_lines.append(plot_path(ax, s2mt_x_l, s2mt_y_l, label='S2MT', cmap_name='Purples'))
    path_lines.append(plot_path(ax, arc_x_l, arc_y_l, label='ARC', cmap_name='Oranges'))
    path_lines.append(plot_path(ax, apf_x_l, apf_y_l, label='APF', cmap_name='Greys'))
    path_lines.append(plot_path(ax, none_x_l, none_y_l, label='None', cmap_name='Purples'))

    # Remove any None values if paths are empty
    path_lines = [line for line in path_lines if line is not None]

    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Path {path_index} in Physical Space')

    # Create a unique legend
    handles, labels = ax.get_legend_handles_labels()
    if path_lines:
        handles.extend(path_lines)
        labels.extend([line.get_label() for line in path_lines])
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_aspect('equal', 'box')
    ax.grid(True)

    # Adjust plot boundaries
    all_x = [p[0] for p in physical_space.border]
    all_y = [p[1] for p in physical_space.border]
    for obstacle in physical_space.obstacle_list:
        all_x.extend([p[0] for p in obstacle])
        all_y.extend([p[1] for p in obstacle])
    all_x.extend(s2c_x_l + s2o_x_l + srl_x_l)
    all_y.extend(s2c_y_l + s2o_y_l + srl_y_l)

    margin = 1
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Save the plot
    output_dir = "./Plots/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}path_{path_index}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    # Load configuration
    config_path = "./utils/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create physical space
    physical_space = Space(
        border=config["border_phys"],
        raw_obstacle_list=config["obstacles_phys"],
        virtual_border=config["border_virt"],
        virtual_obstacle_list=config["obstacles_virt"]
    )

    # Load paths
    path_path = "./Dataset/eval_100.npy"
    path = np.load(path_path, allow_pickle=True)
    if len(path) == 0:
        print("Path file is empty.")
        return

    # Initialize environment objects
    env_s2c = PassiveHapticsEnv(config, "s2c")
    env_s2o = PassiveHapticsEnv(config, "s2o")
    env_srl = PassiveHapticsEnv(config, "srl")
    env_s2mt = PassiveHapticsEnv(config, "s2mt")
    env_arc = PassiveHapticsEnv(config, "arc")
    env_apf = PassiveHapticsEnv(config, "apf")
    env_none = PassiveHapticsEnv(config, "none")

    s2c_x_l, s2c_y_l, s2c_collide_num = [], [], 0
    s2o_x_l, s2o_y_l, s2o_collide_num = [], [], 0
    srl_x_l, srl_y_l, srl_collide_num = [], [], 0
    s2mt_x_l, s2mt_y_l, s2mt_collide_num = [], [], 0
    arc_x_l, arc_y_l, arc_collide_num = [], [], 0
    apf_x_l, apf_y_l, apf_collide_num = [], [], 0
    none_x_l, none_y_l, none_collide_num = [], [], 0
    
    s2c_collide_list = []
    s2o_collide_list = []
    srl_collide_list = []
    s2mt_collide_list = []
    arc_collide_list = []
    apf_collide_list = []
    none_collide_list = []
    
    # Process each path
    for i in range(len(path[:10])):
        current_path = path[i].tolist()
        s2c_x_l, s2c_y_l, s2c_collide_num = env_s2c.step_specific_path(current_path)
        s2o_x_l, s2o_y_l, s2o_collide_num = env_s2o.step_specific_path(current_path)
        srl_x_l, srl_y_l, srl_collide_num = env_srl.step_specific_path(current_path)
        s2mt_x_l, s2mt_y_l, s2mt_collide_num = env_s2mt.step_specific_path(current_path)
        arc_x_l, arc_y_l, arc_collide_num = env_arc.step_specific_path(current_path)
        apf_x_l, apf_y_l, apf_collide_num = env_apf.step_specific_path(current_path)
        none_x_l, none_y_l, none_collide_num = env_none.step_specific_path(current_path)

        plot_fig(physical_space, s2c_x_l, s2c_y_l, s2o_x_l, s2o_y_l, srl_x_l, srl_y_l, s2mt_x_l, s2mt_y_l, arc_x_l, arc_y_l, apf_x_l, apf_y_l,none_x_l, none_y_l, i)
        
        print("s2c_collide_num: ", s2c_collide_num, "s2o_collide_num: ", s2o_collide_num, "srl_collide_num: ", srl_collide_num, "s2mt_collide_num: ", s2mt_collide_num, "arc_collide_num: ", arc_collide_num, "apf_collide_num: ", apf_collide_num, "none_collide_num: ", none_collide_num)
        
        s2c_collide_list.append(s2c_collide_num)
        s2o_collide_list.append(s2o_collide_num)
        srl_collide_list.append(srl_collide_num)
        s2mt_collide_list.append(s2mt_collide_num)
        arc_collide_list.append(arc_collide_num)
        apf_collide_list.append(apf_collide_num)
        none_collide_list.append(none_collide_num)

    print("All path plots have been saved to './Plots/' directory.")
    
    # calculate the collision mean, std, and max, and save to a file
     # Calculate collision statistics
    collision_data = {
        "S2C": {
            "mean": float(np.mean(s2c_collide_list)),
            "std": float(np.std(s2c_collide_list)),
            "max": int(np.max(s2c_collide_list))
        },
        "S2O": {
            "mean": float(np.mean(s2o_collide_list)),
            "std": float(np.std(s2o_collide_list)),
            "max": int(np.max(s2o_collide_list))
        },
        "SRL": {
            "mean": float(np.mean(srl_collide_list)),
            "std": float(np.std(srl_collide_list)),
            "max": int(np.max(srl_collide_list))
        },
        "S2MT": {
            "mean": float(np.mean(s2mt_collide_list)),
            "std": float(np.std(s2mt_collide_list)),
            "max": int(np.max(s2mt_collide_list))
        },
        "ARC": {
            "mean": float(np.mean(arc_collide_list)),
            "std": float(np.std(arc_collide_list)),
            "max": int(np.max(arc_collide_list))
        },
        "APF": {
            "mean": float(np.mean(apf_collide_list)),
            "std": float(np.std(apf_collide_list)),
            "max": int(np.max(apf_collide_list))
        },
        "None": {
            "mean": float(np.mean(none_collide_list)),
            "std": float(np.std(none_collide_list)),
            "max": int(np.max(none_collide_list))
        }
    }

    # Save collision data to JSON
    collision_data_path = "./Plots/collision_data.json"
    with open(collision_data_path, 'w') as f:
        json.dump(collision_data, f, indent=4)
    print(f"Collision data has been saved to '{collision_data_path}'.")


if __name__ == '__main__':
    main()

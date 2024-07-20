import time
import logging

import numpy as np
import matplotlib.pyplot as plt 

import utils
import unit_metric_computers as umc

"""
Compare the same units between environments.

First requires environment-specific and model layer unit charts 
    by running `inspect_model_units.py`

Second, here we visualize how same unit's 2D heatmap differs in different envs.
"""


def _plot_between_envs_unit_heatmaps(config_1, config_2, experiment, moving_trajectory):
    config_1 = utils.load_config(config_1)
    results_path_1 = utils.load_results_path(
        config=config_1,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    fpath = f'{results_path_1}/unit_chart.npy'
    unit_chart_env_1 = np.load(fpath, allow_pickle=True)

    config_2 = utils.load_config(config_2)
    results_path_2 = utils.load_results_path(
        config=config_2,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    fpath = f'{results_path_2}/unit_chart.npy'
    unit_chart_env_2 = np.load(fpath, allow_pickle=True)

    unit_indices_by_types_1 = umc._unit_chart_type_classification(unit_chart_env_1)
    unit_indices_by_types_2 = umc._unit_chart_type_classification(unit_chart_env_2)
    exclusive_place_cells_indices_1 = unit_indices_by_types_1['exclusive_place_cells_indices']
    exclusive_place_cells_indices_2 = unit_indices_by_types_2['exclusive_place_cells_indices']
    print(f"Num of exclusive place cells in env1: {len(exclusive_place_cells_indices_1)}")
    print(f"Num of exclusive place cells in env2: {len(exclusive_place_cells_indices_2)}")

    mutual_place_cells_indices = list(set(exclusive_place_cells_indices_1) & set(exclusive_place_cells_indices_2))
    print(f"Num of mutual place cells: {len(mutual_place_cells_indices)}")

    either_place_cells_indices = list(set(exclusive_place_cells_indices_1) | set(exclusive_place_cells_indices_2))
    print(f"Num of either place cells: {len(either_place_cells_indices)}")

    fig = plt.figure(figsize=(15, 600))
    for row_index, unit_index in enumerate(either_place_cells_indices):
        unit_heatmap_1 = unit_chart_env_1[unit_index, 12]
        unit_heatmap_2 = unit_chart_env_2[unit_index, 12]
        
        # left
        ax = fig.add_subplot(len(either_place_cells_indices), 2, row_index*2+1)
        if unit_index in exclusive_place_cells_indices_1:
            ax.set_title("Place Cell")
        if type(unit_heatmap_1) == int: # dead unit, do not plot
            continue
        ax.imshow(unit_heatmap_1, cmap='jet', interpolation='nearest', label='0 degrees')
        ax.set_xticks([])
        ax.set_yticks([])

        # right
        ax = fig.add_subplot(len(either_place_cells_indices), 2, row_index*2+2)
        if unit_index in exclusive_place_cells_indices_2:
            ax.set_title("Place Cell")
        if type(unit_heatmap_2) == int:
            continue
        ax.imshow(unit_heatmap_2, cmap='jet', interpolation='nearest', label='+45 degrees')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("remapping.png")
        

def main(config_1, config_2, experiment, moving_trajectory):
    _plot_between_envs_unit_heatmaps(config_1, config_2, experiment, moving_trajectory)


if __name__ == '__main__':
    start_time = time.time()
    logging_level = 'info'
    if logging_level == 'info':
        logging.basicConfig(level=logging.INFO)
    elif logging_level == 'debug':
        logging.basicConfig(level=logging.DEBUG)

    # ======================================== #
    experiment = 'unit_chart'
    moving_trajectory = 'uniform'
    config_1 = "env28_r24_2d_vgg16_fc2"
    config_2 = "env37_r24_2d_vgg16_fc2"

    main(config_1, config_2, experiment, moving_trajectory)

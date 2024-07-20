import time
import logging

import numpy as np
import matplotlib.pyplot as plt 

import utils

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


    fig = plt.figure(figsize=(15, 600))

    # TEMP:
    unit_chart_env_1 = unit_chart_env_1[:400]
    print(unit_chart_env_1.shape)

    for row_index, unit_index in enumerate(range(unit_chart_env_1.shape[0])):
        unit_heatmap_1 = unit_chart_env_1[unit_index, 12] # flattened
        unit_heatmap_2 = unit_chart_env_2[unit_index, 12] # flattened
        
        if type(unit_heatmap_1) == int or type(unit_heatmap_2) == int:
            print()
            continue

        ax = fig.add_subplot(unit_chart_env_1.shape[0], 2, row_index*2+1)
        ax.imshow(unit_heatmap_1, cmap='jet', interpolation='nearest', label='0 degrees')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(unit_chart_env_1.shape[0], 2, row_index*2+2)
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

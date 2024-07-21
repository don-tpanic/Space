import numpy as np
import matplotlib.pyplot as plt
import logging
import time

import utils
import unit_metric_computers as umc


def _plot_between_envs_unit_heatmaps(configs, experiment, moving_trajectory):
    unit_charts = []
    unit_indices_by_types = []
    exclusive_place_cells_indices = []

    for config in configs:
        config = utils.load_config(config)
        results_path = utils.load_results_path(
            config=config,
            experiment=experiment,
            moving_trajectory=moving_trajectory,
        )
        fpath = f'{results_path}/unit_chart.npy'
        unit_chart = np.load(fpath, allow_pickle=True)
        unit_charts.append(unit_chart)
        
        indices = umc._unit_chart_type_classification(unit_chart)
        unit_indices_by_types.append(indices)
        exclusive_place_cells_indices.append(indices['exclusive_place_cells_indices'])
        
        print(f"Num of exclusive place cells in {config['config_version']}: {len(indices['exclusive_place_cells_indices'])}")

    mutual_place_cells_indices = set.intersection(*map(set, exclusive_place_cells_indices))
    print(f"Num of mutual place cells: {len(mutual_place_cells_indices)}")

    either_place_cells_indices = set.union(*map(set, exclusive_place_cells_indices))
    print(f"Num of either place cells: {len(either_place_cells_indices)}")

    fig = plt.figure(figsize=(5, 300))
    for row_index, unit_index in enumerate(either_place_cells_indices):
        for col_index, config in enumerate(configs):
            unit_chart = unit_charts[col_index]
            unit_heatmap = unit_chart[unit_index, 12]
            ax = fig.add_subplot(len(either_place_cells_indices), len(configs), row_index * len(configs) + col_index + 1)
            if unit_index in exclusive_place_cells_indices[col_index]:
                ax.set_title("Place Cell")
            if isinstance(unit_heatmap, np.ndarray):
                ax.imshow(unit_heatmap, cmap='jet', interpolation='nearest')
            else:
                ax.set_title("Dead Unit")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("remapping.png")


def main(configs, experiment, moving_trajectory):
    _plot_between_envs_unit_heatmaps(configs, experiment, moving_trajectory)


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
    configs = [
        "env28_r24_2d_vgg16_fc2",
        "env37_r24_2d_vgg16_fc2",
        "env38_r24_2d_vgg16_fc2",
    ]

    main(configs, experiment, moving_trajectory)
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
    plt.savefig(f"{figs_dir}/changes_in_heatmaps.png")


def _plot_between_envs_unit_types_change(config_version_1, config_version_2, experiment, moving_trajectory):
    """
    For two contrasting envs, each env's units follow the following types:
    
    env_1 = {
        'n_dead_units': n_dead_units,
        'place_border_direction_cells_indices': place_border_direction_cells_indices,
        'place_and_border_not_direction_cells_indices': place_and_border_not_direction_cells_indices,
        'place_and_direction_not_border_cells_indices': place_and_direction_not_border_cells_indices,
        'border_and_direction_not_place_cells_indices': border_and_direction_not_place_cells_indices,
        'exclusive_place_cells_indices': exclusive_place_cells_indices,
        'exclusive_border_cells_indices': exclusive_border_cells_indices,
        'exclusive_direction_cells_indices': exclusive_direction_cells_indices,
    }

    env_2 = {
        ...
    }

    Based on `exclusive_place_cells_indices` of env_1, compare env_2
    to find:
        1. Number of units belong to `n_dead_units`
        2. Number of units belong to `place_border_direction_cells_indices`
        3. Number of units belong to `place_and_border_not_direction_cells_indices`
        4. Number of units belong to `place_and_direction_not_border_cells_indices`
        5. Number of units belong to `border_and_direction_not_place_cells_indices`
        6. Number of units belong to `exclusive_place_cells_indices`
    """
    config_1 = utils.load_config(config_version_1)
    results_path_1 = utils.load_results_path(
        config=config_1,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    fpath_1 = f'{results_path_1}/unit_chart.npy'
    unit_chart_1 = np.load(fpath_1, allow_pickle=True)
    indices_1 = umc._unit_chart_type_classification(unit_chart_1)

    config_2 = utils.load_config(config_version_2)
    results_path_2 = utils.load_results_path(
        config=config_2,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    fpath_2 = f'{results_path_2}/unit_chart.npy'
    unit_chart_2 = np.load(fpath_2, allow_pickle=True)
    indices_2 = umc._unit_chart_type_classification(unit_chart_2)

    exclusive_place_cells_indices_1 = indices_1['exclusive_place_cells_indices']

    # Initialize a dictionary to store the results
    results = {
        'dead_units_indices': 0,
        'place_border_direction_cells': 0,
        'place_and_border_not_direction_cells': 0,
        'place_and_direction_not_border_cells': 0,
        'border_and_direction_not_place_cells': 0,
        'exclusive_place_cells': 0
    }

    # Iterate through the exclusive place cells from env_1
    for idx in exclusive_place_cells_indices_1:
        if idx in indices_2['dead_units_indices']:
            results['dead_units_indices'] += 1
        elif idx in indices_2['place_border_direction_cells_indices']:
            results['place_border_direction_cells'] += 1
        elif idx in indices_2['place_and_border_not_direction_cells_indices']:
            results['place_and_border_not_direction_cells'] += 1
        elif idx in indices_2['place_and_direction_not_border_cells_indices']:
            results['place_and_direction_not_border_cells'] += 1
        elif idx in indices_2['border_and_direction_not_place_cells_indices']:
            results['border_and_direction_not_place_cells'] += 1
        elif idx in indices_2['exclusive_place_cells_indices']:
            results['exclusive_place_cells'] += 1
    
    # Calculate percentages
    total_cells = len(exclusive_place_cells_indices_1)
    percentages = {key: (value / total_cells) * 100 for key, value in results.items()}

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(percentages.keys(), percentages.values())
    plt.title(f"Change in Unit Types from {config_version_1} to {config_version_2}")
    plt.xlabel("Unit Types")
    plt.ylabel("Percentage of Units")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    plot_path = f'{figs_dir}/unit_type_change_plot_{config_version_1}-{config_version_2}.png'
    plt.savefig(plot_path)


def main(configs, experiment, moving_trajectory):
    _plot_between_envs_unit_heatmaps(configs, experiment, moving_trajectory)
    _plot_between_envs_unit_types_change(configs[0], configs[2], experiment, moving_trajectory)


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
    figs_dir = "figs/remapping"

    main(configs, experiment, moving_trajectory)
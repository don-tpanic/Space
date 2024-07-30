import time
import math

import numpy as np
import matplotlib.pyplot as plt
import logging

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


def _plot_between_envs_unit_types_change(configs, experiment, moving_trajectory):
    """
    For two contrasting envs, compare how unit types change from env_1 to env_2.
    For each unit type in env_1, plot the percentage of units that have changed to each type in env_2.
    """
    plt.rcParams.update({'font.size': 22,})

    for config_version_1 in configs:
        for config_version_2 in configs:
            if config_version_1 == config_version_2:
                continue

            config_1 = utils.load_config(config_version_1)
            results_path_1 = utils.load_results_path(
                config=config_1,
                experiment=experiment,
                moving_trajectory=moving_trajectory,
            )
            fpath_1 = f'{results_path_1}/unit_chart.npy'
            unit_chart_1 = np.load(fpath_1, allow_pickle=True)
            indices_1 = umc._unit_chart_type_classification(unit_chart_1)

            # Get original unit type percentages
            total_cells_original = sum(len(indices_1[key]) for key in indices_1)
            type_proportions_original = {}
            for unit_type, index_list in indices_1.items():
                proportion = len(index_list) / total_cells_original * 100
                if proportion > 0:
                    type_proportions_original[unit_types[unit_type]] = proportion


            config_2 = utils.load_config(config_version_2)
            results_path_2 = utils.load_results_path(
                config=config_2,
                experiment=experiment,
                moving_trajectory=moving_trajectory,
            )
            fpath_2 = f'{results_path_2}/unit_chart.npy'
            unit_chart_2 = np.load(fpath_2, allow_pickle=True)
            indices_2 = umc._unit_chart_type_classification(unit_chart_2)

            type_change_percentages = {}
            # e.g., {"Inactive": {"Inactive": 1, "P+B+D": 2, "P+B": 3, "P+D": 4, "B+D": 5, "P": 6}}

            for i, unit_type_1 in enumerate(unit_types.keys()):
                results = {}
                # e.g.  = {'Inactive': 1, 'P+B+D': 2, 'P+B': 3, 'P+D': 4, 'B+D': 5, 'P': 6}
                
                # Count how many units of type_1 changed to each type in env_2
                for idx in indices_1[unit_type_1]:
                    for unit_type_2 in unit_types:
                        if idx in indices_2[unit_type_2]:
                            unit_type_2_print = unit_types[unit_type_2]
                            results[unit_type_2_print] = results.get(unit_type_2_print, 0) + 1
                            break  # once found, can safety break because each unit can only change to one type
                
                # Calculate percentages
                total_cells = len(indices_1[unit_type_1])
                if total_cells > 0:
                    percentages = {key: (value / total_cells) * 100 for key, value in results.items()}
                
                    # Store the percentages
                    type_change_percentages[unit_types[unit_type_1]] = percentages

            # Determine the number of rows and columns
            import math
            num_subplots = len(type_change_percentages)
            num_cols = 4  # fixed for nice visualization
            num_rows = math.ceil(num_subplots / num_cols)

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 6 * num_rows))

            # Flatten axes array in case of multiple rows
            axes = axes.flatten()

            # Iterate through data and plot
            for i, (unit_type, percentages) in enumerate(type_change_percentages.items()):
                ax = axes[i]
                labels = percentages.keys()
                colors = [type2color[label] for label in labels]
                ax.pie(
                    percentages.values(), 
                    labels=labels, 
                    autopct=lambda p: '{:.1f}'.format(round(p, 1)) if p > 0 else '',
                    explode=[0.1]*len(labels),
                    startangle=0,
                    colors=colors
                )
                ax.set_title(f"Original: {unit_type}, %= {type_proportions_original[unit_type]:.2f}")

            # Remove any empty subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # Save the plot
            plt.tight_layout()
            plot_path = f'{figs_dir}/unit_type_change_plot_{config_version_1}-{config_version_2}.png'
            plt.savefig(plot_path)
            plt.close()


def _plot_each_env_cell_type_proportions(configs, experiment, moving_trajectory):
    """
    For each given config, plot the proportion of different cell types as a pie chart.
    """
    plt.rcParams.update({'font.size': 22,})

    num_configs = len(configs)
    num_cols = 5
    num_rows = math.ceil(num_configs / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 8 * num_rows))
    axes = axes.flatten() if num_configs > 1 else [axes]

    for i, config_version in enumerate(configs):
        config = utils.load_config(config_version)
        results_path = utils.load_results_path(
            config=config,
            experiment=experiment,
            moving_trajectory=moving_trajectory,
        )
        fpath = f'{results_path}/unit_chart.npy'
        unit_chart = np.load(fpath, allow_pickle=True)
        indices = umc._unit_chart_type_classification(unit_chart)

        type_proportions = {}
        total_units = sum(len(indices[key]) for key in indices)
        print(f"Total units in {config_version}: {total_units}")

        for unit_type, index_list in indices.items():
            proportion = len(index_list) / total_units * 100
            if proportion > 0:
                type_proportions[unit_types[unit_type]] = proportion

        ax = axes[i]
        labels = type_proportions.keys()
        colors = [type2color[label] for label in labels]

        ax.pie(
            type_proportions.values(), 
            labels=labels, 
            autopct=lambda p: '{:.1f}'.format(round(p, 1)) if p > 0 else '',
            startangle=90,
            colors=colors,
            explode=[0.1]*len(labels)
        )

        ax.set_title(f"{envs2changes[config_version]}")

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plot_path = f'{figs_dir}/each_env_cell_type_proportions.png'
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")


def main(configs, experiment, moving_trajectory):
    _plot_between_envs_unit_heatmaps(configs, experiment, moving_trajectory)
    _plot_between_envs_unit_types_change(configs, experiment, moving_trajectory)
    _plot_each_env_cell_type_proportions(configs, experiment, moving_trajectory)


if __name__ == '__main__':
    envs2changes = {
        "env28run2_r24_2d_vgg16_fc2": "original",
        "env37_r24_2d_vgg16_fc2": "45 deg",
        "env38_r24_2d_vgg16_fc2": "90 deg",
        "env39_r24_2d_vgg16_fc2": "many items changes",
        "env40_r24_2d_vgg16_fc2": "one item change",
    }

    unit_types = {
        'dead_units_indices': 'Inactive',
        'place_border_direction_cells_indices': 'P+B+D',
        'place_and_border_not_direction_cells_indices': 'P+B',
        'place_and_direction_not_border_cells_indices': 'P+D',
        'border_and_direction_not_place_cells_indices': 'B+D',
        'exclusive_place_cells_indices': 'P',
        'exclusive_border_cells_indices': 'B',
        'exclusive_direction_cells_indices': 'D',
        'active_no_type_indices': 'No Type'
    }

    type2color = {
        "P": plt.cm.Pastel1.colors[1],
        "B": plt.cm.Pastel1.colors[0],
        "D": plt.cm.Pastel1.colors[2],
        "P+B": plt.cm.Pastel1.colors[3],
        "P+D": plt.cm.Pastel1.colors[4],
        "B+D": plt.cm.Pastel1.colors[5],
        "P+B+D": plt.cm.Pastel1.colors[6],
        "No Type": plt.cm.Pastel1.colors[7],
        "Inactive": "grey"
    }

    start_time = time.time()
    logging_level = 'info'
    if logging_level == 'info':
        logging.basicConfig(level=logging.INFO)
    elif logging_level == 'debug':
        logging.basicConfig(level=logging.DEBUG)

    # ======================================== #
    experiment = 'unit_chart'
    moving_trajectory = 'uniform'
    configs = list(envs2changes.keys())
    figs_dir = "figs/remapping"

    main(configs, experiment, moving_trajectory)
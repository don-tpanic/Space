import time
import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

import utils
import unit_metric_computers as umc


def _plot_between_envs_unit_heatmaps(configs, experiment, moving_trajectory):
    """
    For same units between two envs, plot the heatmaps side by side.

    Right now we only consider units that are place cells in at least one of the envs.
    """
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

    for config_i, config_version_1 in enumerate(configs):
        for config_j, config_version_2 in enumerate(configs):
            if config_i >= config_j:
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
            num_cols = 3  # fixed for nice visualization
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
                    autopct=lambda p: '{:.1f}'.format(round(p, 1)) if p > 0. else '',
                    explode=[0.1]*len(labels),
                    startangle=0,
                    colors=colors
                )
                ax.set_title(f"Originally: {unit_type}, ({type_proportions_original[unit_type]:.2f}%)", 
                            fontsize=22, fontweight='bold')

            # Remove any empty subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # Save the plot
            plt.tight_layout()
            plot_path = f'{figs_dir}/unit_type_change_plot_{config_version_1}-{config_version_2}.pdf'
            print(f"Plot saved to {plot_path}")
            plt.savefig(plot_path)
            plt.close()


def _plot_each_env_cell_type_proportions(configs, experiment, moving_trajectory):
    """
    For each given env, plot the proportion of different cell types as a pie chart.
    """
    plt.rcParams.update({'font.size': 22,})

    num_configs = len(configs)
    num_cols = num_configs
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

        ax.set_title(f"{envs2changes[config_version]}", fontsize=22, fontweight='bold')

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plot_path = f'{figs_dir}/each_env_cell_type_proportions_{model}_{layer}.pdf'
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")


def _plot_between_envs_unit_type_P_change(configs, experiment, moving_trajectory):
    """
    Focus on exclusive P cells as part of the remapping analysis.

    Between two envs, quantify:
        0. The percentage of exclusive P cells in each env
        1. How many P cells stay as P cells (cell-to-cell)
        2. How many P cells maintain number of place fields (cell-to-cell)
        3. Overall, change in number of place fields
    """
    plt.rcParams.update({'font.size': 22,})

    results = []
    for i, config_version_1 in enumerate(configs):
        for j, config_version_2 in enumerate(configs):
            if i >= j:
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

            config_2 = utils.load_config(config_version_2)
            results_path_2 = utils.load_results_path(
                config=config_2,
                experiment=experiment,
                moving_trajectory=moving_trajectory,
            )
            fpath_2 = f'{results_path_2}/unit_chart.npy'
            unit_chart_2 = np.load(fpath_2, allow_pickle=True)
            indices_2 = umc._unit_chart_type_classification(unit_chart_2)

            # Exclusive P cell percentages
            p_proportion_1 = len(indices_1['exclusive_place_cells_indices']) / sum(len(v) for v in indices_1.values()) * 100
            p_proportion_2 = len(indices_2['exclusive_place_cells_indices']) / sum(len(v) for v in indices_2.values()) * 100

            # Exclusive P cells stay as P cells
            p_cells_1 = set(indices_1['exclusive_place_cells_indices'])
            p_cells_2 = set(indices_2['exclusive_place_cells_indices'])
            num_p_cells_stay_p = len(p_cells_1.intersection(p_cells_2))
            p_cells_stay_p_proportion = num_p_cells_stay_p / len(p_cells_1) * 100 if len(p_cells_1) > 0 else 0

            # Exclusive P cells maintaining number of place fields
            num_p_cells_maintain_nfields = 0
            num_fields_1 = []
            num_fields_2 = []
            for p_cell in p_cells_1.intersection(p_cells_2):
                num_fields_1.append(unit_chart_1[p_cell, 1][0])
                num_fields_2.append(unit_chart_2[p_cell, 1][0])
                if np.array_equal(unit_chart_1[p_cell, 1], unit_chart_2[p_cell, 1]):
                    num_p_cells_maintain_nfields += 1

            p_cells_maintain_nfields_proportion = num_p_cells_maintain_nfields / num_p_cells_stay_p * 100 if num_p_cells_stay_p > 0 else 0

            # Store the results for plotting
            results.append({
                'config_pair': (config_version_1, config_version_2),
                'p_proportion_1': p_proportion_1,
                'p_proportion_2': p_proportion_2,
                'p_cells_stay_p_proportion': p_cells_stay_p_proportion,
                'p_cells_maintain_nfields_proportion': p_cells_maintain_nfields_proportion,
                'num_fields_1': num_fields_1,
                'num_fields_2': num_fields_2,
            })

    # Plotting
    for result in results:
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        config_pair = result['config_pair']
        labels = [envs2changes[config_pair[0]], envs2changes[config_pair[1]]]

        # Plot exclusive P cell percentages
        pie1 = axes[0, 0].pie([result['p_proportion_1'], 100-result['p_proportion_1']], 
                    colors=['#1f77b4', '#d3d3d3'], 
                    labels=[labels[0], ''],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(width=0.4),
                    radius=0.8,
                    center=(-0.8, 0))  # Move the first pie chart to the left

        pie2 = axes[0, 0].pie([result['p_proportion_2'], 100-result['p_proportion_2']], 
                            colors=['#ff7f0e', '#d3d3d3'], 
                            labels=[labels[1], ''],
                            autopct='%1.1f%%',
                            startangle=90,
                            wedgeprops=dict(width=0.4),
                            radius=0.8,
                            center=(0.8, 0))  # Move the second pie chart to the right

        # Adjust the subplot layout
        axes[0, 0].set_xlim(-1.2, 1.2)
        axes[0, 0].set_title("Exclusive P Cell Percentages")

        # Plot percentage of P cells that stay P cells
        axes[0, 1].pie(
            [result['p_cells_stay_p_proportion'], 100 - result['p_cells_stay_p_proportion']],
            labels=['Stayed P', 'Changed'],
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.1, 0],
            colors=['#84B7CA', '#D0C146']
        )
        axes[0, 1].set_title("P Cells Staying as P Cells")

        # Plot percentage of P cells maintaining number of place fields
        axes[1, 0].pie(
            [result['p_cells_maintain_nfields_proportion'], 100 - result['p_cells_maintain_nfields_proportion']],
            labels=['Maintained Fields', 'Changed Fields'],
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.1, 0],
            colors=['#763C4B', '#FCA476']
        )
        axes[1, 0].set_title("P Cells Maintaining Number of Place Fields")

        # Plot number of place fields
        sns.kdeplot(result['num_fields_1'], ax=axes[1, 1], label=labels[0], color='#1f77b4')
        sns.kdeplot(result['num_fields_2'], ax=axes[1, 1], label=labels[1], color='#ff7f0e')
        axes[1, 1].set_title("Number of Place Fields")
        axes[1, 1].legend()
        axes[1, 1].spines['top'].set_visible(False)
        axes[1, 1].spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{figs_dir}/P_cell_change_{config_pair[0]}-{config_pair[1]}.pdf")
        plt.close()


def _plot_between_envs_unit_type_P_rotation(configs, experiment, moving_trajectory):
    """
    Focus on exclusive P cells as part of the remapping analysis.

    Between two envs, quantify:
        1. For those P cells maintain the same properties (i.e. num of fields),
            extract their mean angle wrt center of the area, compute the abs difference in angle,
            and plot the distribution of the differences.

        2. For each pair of envs, for the same cell, we also plot the heatmaps.
    """
    plt.rcParams.update({'font.size': 22,})

    angles_before_and_after = {}  
    for i, config_version_1 in enumerate(configs):
        for j, config_version_2 in enumerate(configs):
            if i >= j:
                continue
            angles_before_and_after[(config_version_1, config_version_2)] = {
                "angles": [],
                "heatmaps_1": [],
                "heatmaps_2": []
            }

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

            # Check corresponding p cells in each env, and only compute diff in angle
            # if both have the same number of fields
            p_cells_1 = set(indices_1['exclusive_place_cells_indices'])
            p_cells_2 = set(indices_2['exclusive_place_cells_indices'])

            for p_cell in p_cells_1.intersection(p_cells_2):
                if np.array_equal(unit_chart_1[p_cell, 1], unit_chart_2[p_cell, 1]):
                    angle_1 = unit_chart_1[p_cell, 13]
                    angle_2 = unit_chart_2[p_cell, 13]
                    angle_abs_diff = np.abs(angle_2 - angle_1)

                    # Collect angle differences
                    angles_before_and_after[(config_version_1, config_version_2)]["angles"].append(angle_abs_diff)

                    # Collect heatmaps for visualization
                    angles_before_and_after[(config_version_1, config_version_2)]["heatmaps_1"].append(unit_chart_1[p_cell, 12])
                    angles_before_and_after[(config_version_1, config_version_2)]["heatmaps_2"].append(unit_chart_2[p_cell, 12])

    # Plotting angle differences distribution
    for config_pair, values in angles_before_and_after.items():
        num_p_cells = len(values["angles"])
        print(f"[Check] Num of P cells in {config_pair[0]}-{config_pair[1]}: {num_p_cells}")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        labels = [envs2changes[config_pair[0]], envs2changes[config_pair[1]]]

        sns.kdeplot(values['angles'], ax=ax, color='#1f77b4')
        ax.hist(values['angles'], bins=20, density=True, alpha=0.5, color='#1f77b4')
        ax.set_title("Angle Difference Between P Cells")
        ax.set_xlabel("Angle Difference (degrees)")
        ax.set_ylabel("Density")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{figs_dir}/P_cell_rotation_angle_diff_{config_pair[0]}-{config_pair[1]}.png")
        plt.close()

        fig, axes = plt.subplots(num_p_cells, 2, figsize=(10, 2*num_p_cells))
        for i in range(num_p_cells):
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]

            ax1.imshow(values['heatmaps_1'][i], cmap='jet', interpolation='nearest')
            ax1.set_title(labels[0], fontsize=10)
            ax1.set_xticks([])
            ax1.set_yticks([])

            ax2.imshow(values['heatmaps_2'][i], cmap='jet', interpolation='nearest')
            ax2.set_title(f"{labels[1]}: diff={values['angles'][i]:.1f} deg", fontsize=10)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f"{figs_dir}/P_cell_rotation_heatmap_diff_{config_pair[0]}-{config_pair[1]}.pdf")
        plt.close()


def _plot_between_envs_unit_type_PD_rotation(configs, experiment, moving_trajectory):
    """
    Focus on P+D cells as part of the remapping analysis.

    Between two envs, quantify:
        1. For those P+D cells maintain the same properties (i.e. num of fields),
            extract their mean angle wrt center of the area, compute the abs difference in angle,
            and plot the distribution of the differences.

        2. For each pair of envs, for the same cell, we also plot the heatmaps.
    """
    plt.rcParams.update({'font.size': 22,})

    angles_before_and_after = {}  
    for i, config_version_1 in enumerate(configs):
        for j, config_version_2 in enumerate(configs):
            if i >= j:
                continue
            angles_before_and_after[(config_version_1, config_version_2)] = {
                "angles": [],
                "heatmaps_1": [],
                "heatmaps_2": []
            }

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

            # Check corresponding p cells in each env, and only compute diff in angle
            # if both have the same number of fields
            pd_cells_1 = set(indices_1['place_and_direction_not_border_cells_indices'])
            pd_cells_2 = set(indices_2['place_and_direction_not_border_cells_indices'])

            for pd_cell in pd_cells_1.intersection(pd_cells_2):
                if np.array_equal(unit_chart_1[pd_cell, 1], unit_chart_2[pd_cell, 1]):
                    angle_1 = unit_chart_1[pd_cell, 13]
                    angle_2 = unit_chart_2[pd_cell, 13]
                    angle_abs_diff = np.abs(angle_2 - angle_1)

                    # Collect angle differences
                    angles_before_and_after[(config_version_1, config_version_2)]["angles"].append(angle_abs_diff)

                    # Collect heatmaps for visualization
                    angles_before_and_after[(config_version_1, config_version_2)]["heatmaps_1"].append(unit_chart_1[pd_cell, 12])
                    angles_before_and_after[(config_version_1, config_version_2)]["heatmaps_2"].append(unit_chart_2[pd_cell, 12])

    # Plotting angle differences distribution
    for config_pair, values in angles_before_and_after.items():
        num_pd_cells = len(values["angles"])
        print(f"[Check] Num of PD cells in {config_pair[0]}-{config_pair[1]}: {num_pd_cells}")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        labels = [envs2changes[config_pair[0]], envs2changes[config_pair[1]]]

        sns.kdeplot(values['angles'], ax=ax, color='#1f77b4')
        ax.hist(values['angles'], bins=20, density=True, alpha=0.5, color='#1f77b4')
        ax.set_title("Angle Difference Between P+D Cells")
        ax.set_xlabel("Angle Difference (degrees)")
        ax.set_ylabel("Density")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{figs_dir}/PD_cell_rotation_angle_diff_{config_pair[0]}-{config_pair[1]}.png")
        plt.close()

        if num_pd_cells > 50:  # Avoid plotting huge fig for now.
            num_pd_cells = 50
            
        fig, axes = plt.subplots(num_pd_cells, 2, figsize=(10, 2*num_pd_cells))
        for i in range(num_pd_cells):
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]

            ax1.imshow(values['heatmaps_1'][i], cmap='jet', interpolation='nearest')
            ax1.set_title(labels[0], fontsize=10)
            ax1.set_xticks([])
            ax1.set_yticks([])

            ax2.imshow(values['heatmaps_2'][i], cmap='jet', interpolation='nearest')
            ax2.set_title(f"{labels[1]}: diff={values['angles'][i]:.1f} deg", fontsize=10)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f"{figs_dir}/PD_cell_rotation_heatmap_diff_{config_pair[0]}-{config_pair[1]}.pdf")
        plt.close()


def _plot_between_envs_any_type_with_n_fields_rotation(configs, experiment, moving_trajectory, n_fields=None):
    """
    Focus on any cell type (with n fields) as part of the remapping analysis.

    Between two envs, quantify:
        1. For those cells maintain the same properties (i.e. has n fields),
            extract their mean angle wrt center of the area, compute the abs difference in angle,
            and plot the distribution of the differences.

        2. For each pair of envs, for the same cell, we also plot the heatmaps.

    
    Args:
        n_fields: 1 or `None` (for any number of fields)
    """
    plt.rcParams.update({'font.size': 22,})

    angles_before_and_after = {}  
    for i, config_version_1 in enumerate(configs):
        for j, config_version_2 in enumerate(configs):
            if i >= j:
                continue
            angles_before_and_after[(config_version_1, config_version_2)] = {
                "angles": [],
                "heatmaps_1": [],
                "heatmaps_2": []
            }

            config_1 = utils.load_config(config_version_1)
            results_path_1 = utils.load_results_path(
                config=config_1,
                experiment=experiment,
                moving_trajectory=moving_trajectory,
            )
            fpath_1 = f'{results_path_1}/unit_chart.npy'
            unit_chart_1 = np.load(fpath_1, allow_pickle=True)

            config_2 = utils.load_config(config_version_2)
            results_path_2 = utils.load_results_path(
                config=config_2,
                experiment=experiment,
                moving_trajectory=moving_trajectory,
            )
            fpath_2 = f'{results_path_2}/unit_chart.npy'
            unit_chart_2 = np.load(fpath_2, allow_pickle=True)

            # Go thru all units and check if they have a single field
            for unit_index in range(unit_chart_1.shape[0]):
                # Make sure the unit is not dead in both envs
                if unit_chart_1[unit_index, 0][0] == 1 and unit_chart_2[unit_index, 0][0] == 1:

                    # For active unit, check if it has `n fields` in both envs
                    if n_fields is None:
                        # Have same number of fields including all non-zero fields
                        if unit_chart_1[unit_index, 1][0] == unit_chart_2[unit_index, 1][0] and \
                            unit_chart_1[unit_index, 1][0] > 0 and unit_chart_2[unit_index, 1][0] > 0:
                            angle_1 = unit_chart_1[unit_index, 13]
                            angle_2 = unit_chart_2[unit_index, 13]
                            angle_abs_diff = np.abs(angle_2 - angle_1)
                            # Collect angle differences
                            angles_before_and_after[(config_version_1, config_version_2)]["angles"].append(angle_abs_diff)

                    else:
                        # Have exactly `n_fields`
                        if unit_chart_1[unit_index, 1][0] == n_fields and unit_chart_2[unit_index, 1][0] == n_fields:
                            angle_1 = unit_chart_1[unit_index, 13]
                            angle_2 = unit_chart_2[unit_index, 13]
                            angle_abs_diff = np.abs(angle_2 - angle_1)
                            # Collect angle differences
                            angles_before_and_after[(config_version_1, config_version_2)]["angles"].append(angle_abs_diff)

    # Plotting angle differences distribution
    for config_pair, values in angles_before_and_after.items():
        num_cells = len(values["angles"])
        print(f"[Check] Num of cells with {n_fields} fields in {config_pair[0]}-{config_pair[1]}: {num_cells}")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        labels = [envs2changes[config_pair[0]], envs2changes[config_pair[1]]]

        sns.kdeplot(values['angles'], ax=ax, color='#1f77b4')
        ax.hist(values['angles'], bins=20, density=True, alpha=0.5, color='#1f77b4')
        # ax.set_title("Angle Difference Between Cells with Single Field")
        if n_fields is None:
            ax.set_title("Angle Difference Between Cells with Any Number of Fields", fontsize=18)
        else:
            ax.set_title(f"Angle Difference Between Cells with {n_fields} Field(s)", fontsize=18)
        ax.set_xlabel("Angle Difference (degrees)")
        ax.set_ylabel("Density")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        if n_fields is None:
            plt.savefig(f"{figs_dir}/any_cell_with_any_fields_rotation_angle_diff_{config_pair[0]}-{config_pair[1]}.pdf")
        else:
            plt.savefig(f"{figs_dir}/any_cell_with_{n_fields}_fields_rotation_angle_diff_{config_pair[0]}-{config_pair[1]}.pdf")
        plt.close()


def main(configs, experiment, moving_trajectory):
    # _plot_between_envs_unit_heatmaps(configs, experiment, moving_trajectory)
    _plot_between_envs_unit_types_change(configs, experiment, moving_trajectory)
    _plot_each_env_cell_type_proportions(configs, experiment, moving_trajectory)
    # _plot_between_envs_unit_type_P_change(configs, experiment, moving_trajectory)
    # _plot_between_envs_unit_type_P_rotation(configs, experiment, moving_trajectory)
    # _plot_between_envs_unit_type_PD_rotation(configs, experiment, moving_trajectory)
    _plot_between_envs_any_type_with_n_fields_rotation(configs[:2], experiment, moving_trajectory, n_fields=1)
    _plot_between_envs_any_type_with_n_fields_rotation(configs[:2], experiment, moving_trajectory, n_fields=None)


if __name__ == '__main__':

    model = 'vgg16'
    layer = 'block4_pool'

    envs2changes = {
        f"env28run2_r24_2d_{model}_{layer}": "original",
        f"env37_r24_2d_{model}_{layer}": "45 deg",
        f"env39_r24_2d_{model}_{layer}": "many item changes",
        f"env40_r24_2d_{model}_{layer}": "one item change",
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
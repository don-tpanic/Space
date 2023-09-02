import os
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

import data
import utils
import models


def _convert_mse_to_physical_unit(mse, error_type):
    """
    Convert the MSE error back to physical sense,
    For location and distance error, we take the square root of MSE,
    which means on average how far off the prediction is from truth
    in terms of Unity units.

    For rotation error, we map the error back to degree by first 
    taking the square root of MSE, which gives 'how many intervals'
    is the prediction off, and since we have 24 intervals out of 360 degrees,
    we can map the error back to degree by multiplying the square root of MSE
    by 360/24.
    """
    if error_type == 'loc' or error_type == 'dist':
        return np.sqrt(mse)
    elif error_type == 'rot':
        return np.sqrt(mse) * 360/24


def decoding_each_model_across_layers_and_sr():
    envs = ['env28_r24']
    env = envs[0]
    movement_mode = '2d'
    sampling_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    random_seeds = [42]
    model_names = ['vgg16', 'vgg16_untrained', 
        'resnet50', 'resnet50_untrained',
        'vit_b16', 'vit_b16_untrained']
    moving_trajectory = 'uniform'
    decoding_model_choice = {'name': 'ridge_regression', 'hparams': 1.0}
    decoding_model_name = decoding_model_choice['name']
    decoding_model_hparams = decoding_model_choice['hparams']
    feature_selection = 'l2'
    error_types = ['loc', 'rot', 'dist']
    tracked_metrics = ['mse', 'ci', 'baseline_predict_mid_mse', 'baseline_predict_random_mse']

    for model_name in model_names:
        output_layers = data.load_model_layers(model_name)

        results_collector = \
            defaultdict(                            # key - error_type
                lambda: defaultdict(                # key - output_layer
                    lambda: defaultdict(list)       # key - metric
                )
            )
                        
        for error_type in error_types:
            if 'loc' in error_type or 'rot' in error_type:
                experiment = 'loc_n_rot'
            elif 'dist' in error_type:
                experiment = 'border_dist'

            for output_layer in output_layers:
                for sampling_rate in sampling_rates:
                    # sampling rate would be the base dimension where 
                    # we accumulate results in a list to plot at once.
                    to_average_over_seeds = defaultdict(list)
                    for random_seed in random_seeds:
                        results_path = \
                            f'results/{env}/{movement_mode}/{moving_trajectory}/'\
                            f'{model_name}/{experiment}/{feature_selection}/'\
                            f'{decoding_model_name}_{decoding_model_hparams}/'\
                            f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
                        results = np.load(f'{results_path}/res.npy', allow_pickle=True).item()[error_type]
                        for metric in tracked_metrics:
                            to_average_over_seeds[metric].append(results[metric])
                    
                    # per metric per output layer 
                    # across sampling rates averaged over seeds
                    for metric in tracked_metrics:
                        # a special case is when metric=='ci' where 
                        # ..res[metric] is a list of 2 elements
                        # so we need to average wrt each element across seeds
                        # and save them back as 2 elements for later plotting.
                        if metric == 'ci':
                            ci_low_avg = np.mean(
                                [ci[0] for ci in to_average_over_seeds[metric]])
                            ci_high_avg = np.mean(
                                [ci[1] for ci in to_average_over_seeds[metric]])
                            avg_res = [ci_low_avg, ci_high_avg]
                        else:
                            avg_res = np.mean(to_average_over_seeds[metric])
                        results_collector[error_type][output_layer][metric].append(avg_res)
        
        # plot collected results.
        fig, axes = plt.subplots(1, len(error_types), figsize=(15, 5))
        for i, error_type in enumerate(error_types):
            for output_layer in output_layers:
                for metric in tracked_metrics:
                    # when metric is about confidence interval, 
                    # instead of plot, we fill_between
                    if metric == 'ci':
                        ci_low = np.array(
                            results_collector[error_type][output_layer][metric])[:, 0]
                        ci_high = np.array(
                            results_collector[error_type][output_layer][metric])[:, 1]
                        
                        # # TEMP
                        # ci_low = _convert_mse_to_physical_unit(ci_low, error_type)
                        # ci_high = _convert_mse_to_physical_unit(ci_high, error_type)

                        axes[i].fill_between(
                            sampling_rates,
                            ci_low,
                            ci_high,
                            alpha=0.3,
                            color='grey',
                        )
                    else:
                        if 'baseline' in metric:
                            # no need to label baseline for each layer
                            # we only going to label baseline when we plot
                            # the last layer.
                            if output_layer == output_layers[-1]:
                                if 'mid' in metric:
                                    label = 'baseline: center'
                                else:
                                    label = 'baseline: random'
                            else:
                                label = None  
                            if 'mid' in metric: 
                                color = 'cyan'
                            else: 
                                color = 'blue'
                        else:
                            # for non-baseline layer performance,
                            # we label each layer and use layer-specific color.
                            label = output_layer
                            if "predictions" in label: label = "logits"
                            color = data.load_envs_dict(model_name, envs)[
                                f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                        
                        # either baseline or non-baseline layer performance,
                        # we always plot them.
                        axes[i].plot(
                            sampling_rates,
                            results_collector[error_type][output_layer][metric],
                            # # TEMP    
                            # _convert_mse_to_physical_unit(
                            #     np.array(results_collector[error_type][output_layer][metric]),
                            #     error_type
                            # ),
                            label=label,
                            color=color,
                            marker='o',
                        )
            axes[i].set_xlabel('Sampling rate')
            axes[i].set_ylabel('Decoding error (MSE)')
            axes[i].set_xticks(sampling_rates)
            axes[i].set_xticklabels(sampling_rates)
            if error_type == 'loc':  title = 'Location Decoding'
            elif error_type == 'rot': title = 'Direction Decoding'
            elif error_type == 'dist': title = 'Distance to Nearest Border Decoding'
            axes[i].set_title(title)
            axes[i].spines.right.set_visible(False)
            axes[i].spines.top.set_visible(False)
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.savefig(f'figs/paper/decoding_{model_name}.png')
        plt.close()


def decoding_all_models_one_layer_one_sr_V1():
    envs = ['env28_r24']
    env = envs[0]
    movement_mode = '2d'
    sampling_rates = [0.3]
    random_seeds = [42]
    model_names = ['vgg16', 'vgg16_untrained', 
        'resnet50', 'resnet50_untrained',
        'vit_b16', 'vit_b16_untrained']
    moving_trajectory = 'uniform'
    decoding_model_choice = {'name': 'ridge_regression', 'hparams': 1.0}
    decoding_model_name = decoding_model_choice['name']
    decoding_model_hparams = decoding_model_choice['hparams']
    feature_selection = 'l2'
    error_types = ['loc', 'rot', 'dist']
    tracked_metrics = ['mse', 'ci', 'baseline_predict_mid_mse', 'baseline_predict_random_mse']


    results_collector = \
        defaultdict(                            # key - error_type
            lambda: defaultdict(                # key - model_name
                lambda: defaultdict(            # key - output_layer
                    lambda: defaultdict(list)   # key - metric
                )
            )
        )
         
    for error_type in error_types:
        if 'loc' in error_type or 'rot' in error_type:
            experiment = 'loc_n_rot'
        elif 'dist' in error_type:
            experiment = 'border_dist'

        for model_name in model_names:
            if 'vgg16' in model_name:  output_layer = 'block5_pool'
            elif 'resnet50' in model_name: output_layer = 'avg_pool'
            elif 'vit' in model_name: output_layer = 'layer_12'

            for sampling_rate in sampling_rates:
                # sampling rate would be the base dimension where 
                # we accumulate results in a list to plot at once.
                to_average_over_seeds = defaultdict(list)
                for random_seed in random_seeds:
                    results_path = \
                        f'results/{env}/{movement_mode}/{moving_trajectory}/'\
                        f'{model_name}/{experiment}/{feature_selection}/'\
                        f'{decoding_model_name}_{decoding_model_hparams}/'\
                        f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
                    results = np.load(f'{results_path}/res.npy', allow_pickle=True).item()[error_type]
                    for metric in tracked_metrics:
                        to_average_over_seeds[metric].append(results[metric])
                
                # per metric per output layer 
                # across sampling rates averaged over seeds
                for metric in tracked_metrics:
                    # a special case is when metric=='ci' where 
                    # ..res[metric] is a list of 2 elements
                    # so we need to average wrt each element across seeds
                    # and save them back as 2 elements for later plotting.
                    if metric == 'ci':
                        ci_low_avg = np.mean(
                            [ci[0] for ci in to_average_over_seeds[metric]])
                        ci_high_avg = np.mean(
                            [ci[1] for ci in to_average_over_seeds[metric]])
                        avg_res = [ci_low_avg, ci_high_avg]
                    else:
                        avg_res = np.mean(to_average_over_seeds[metric])
                    results_collector[error_type][model_name][output_layer][metric].append(avg_res)
    
    # plot collected results.
    fig, axes = plt.subplots(1, len(error_types), figsize=(15, 5))
    for i, error_type in enumerate(error_types):
        for x_i, model_name in enumerate(model_names):
            model_name = model_names[x_i]
            if 'vgg16' in model_name:  output_layer = 'block5_pool'
            elif 'resnet50' in model_name: output_layer = 'avg_pool'
            elif 'vit' in model_name: output_layer = 'layer_12'

            mse = np.array(
                results_collector[error_type][model_name][output_layer]['mse'])
            ci_low = np.array(
                results_collector[error_type][model_name][output_layer]['ci'])[:, 0]
            ci_high = np.array(
                results_collector[error_type][model_name][output_layer]['ci'])[:, 1]
        
            axes[i].errorbar(
                x_i,
                mse,
                yerr=[mse-ci_low, ci_high-mse],
                label=model_name,
                marker='o',
                capsize=5,
            )

            # only highlight if the model is trained 
            # using axvspan
            if 'untrained' not in model_name:
                axes[i].axvspan(x_i-0.5, x_i+0.5, facecolor='grey', alpha=0.3)
                
        # baselines are the same for all models
        # so we only plot them once as plot
        baseline_predict_mid_mse = np.array(
            results_collector[error_type][model_name][output_layer]['baseline_predict_mid_mse'])
        axes[i].plot(
            range(len(model_names)),
            baseline_predict_mid_mse.repeat(len(model_names)),
            label='baseline: center',
            color='cyan',
        )

        baseline_predict_random_mse = np.array(
            results_collector[error_type][model_name][output_layer]['baseline_predict_random_mse'])
        
        axes[i].plot(
            range(len(model_names)),
            baseline_predict_random_mse.repeat(len(model_names)),
            label='baseline: random',
            color='blue',
        )

        # axes[i].set_xlabel('Model')
        axes[i].set_ylabel('Decoding error (MSE)')
        axes[i].set_xticks(range(len(model_names)))
        axes[i].set_xticklabels(model_names, rotation=90)
        if error_type == 'loc':  title = 'Location Decoding'
        elif error_type == 'rot': title = 'Direction Decoding'
        elif error_type == 'dist': title = 'Distance to Nearest Border Decoding'
        axes[i].set_title(title)
        axes[i].spines.right.set_visible(False)
        axes[i].spines.top.set_visible(False)
        
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(f'figs/paper/decoding_across_models.png')
    plt.close()


def decoding_all_models_one_layer_one_sr():
    """
    ref:
        https://stackoverflow.com/questions/5656798/is-there-a-way-to-make-a-discontinuous-axis-in-matplotlib
    """
    envs = ['env28_r24']
    env = envs[0]
    movement_mode = '2d'
    sampling_rates = [0.3]
    random_seeds = [42]
    model_names = ['vgg16', 'vgg16_untrained', 
        'resnet50', 'resnet50_untrained',
        'vit_b16', 'vit_b16_untrained']
    moving_trajectory = 'uniform'
    decoding_model_choice = {'name': 'ridge_regression', 'hparams': 1.0}
    decoding_model_name = decoding_model_choice['name']
    decoding_model_hparams = decoding_model_choice['hparams']
    feature_selection = 'l2'
    error_types = ['loc', 'rot', 'dist']
    tracked_metrics = ['mse', 'ci', 'baseline_predict_mid_mse', 'baseline_predict_random_mse']


    results_collector = \
        defaultdict(                            # key - error_type
            lambda: defaultdict(                # key - model_name
                lambda: defaultdict(            # key - output_layer
                    lambda: defaultdict(list)   # key - metric
                )
            )
        )
         
    for error_type in error_types:
        if 'loc' in error_type or 'rot' in error_type:
            experiment = 'loc_n_rot'
        elif 'dist' in error_type:
            experiment = 'border_dist'

        for model_name in model_names:
            if 'vgg16' in model_name:  output_layer = 'block5_pool'
            elif 'resnet50' in model_name: output_layer = 'avg_pool'
            elif 'vit' in model_name: output_layer = 'layer_12'

            for sampling_rate in sampling_rates:
                # sampling rate would be the base dimension where 
                # we accumulate results in a list to plot at once.
                to_average_over_seeds = defaultdict(list)
                for random_seed in random_seeds:
                    results_path = \
                        f'results/{env}/{movement_mode}/{moving_trajectory}/'\
                        f'{model_name}/{experiment}/{feature_selection}/'\
                        f'{decoding_model_name}_{decoding_model_hparams}/'\
                        f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
                    results = np.load(f'{results_path}/res.npy', allow_pickle=True).item()[error_type]
                    for metric in tracked_metrics:
                        to_average_over_seeds[metric].append(results[metric])
                
                # per metric per output layer 
                # across sampling rates averaged over seeds
                for metric in tracked_metrics:
                    # a special case is when metric=='ci' where 
                    # ..res[metric] is a list of 2 elements
                    # so we need to average wrt each element across seeds
                    # and save them back as 2 elements for later plotting.
                    if metric == 'ci':
                        ci_low_avg = np.mean(
                            [ci[0] for ci in to_average_over_seeds[metric]])
                        ci_high_avg = np.mean(
                            [ci[1] for ci in to_average_over_seeds[metric]])
                        avg_res = [ci_low_avg, ci_high_avg]
                    else:
                        avg_res = np.mean(to_average_over_seeds[metric])
                    results_collector[error_type][model_name][output_layer][metric].append(avg_res)
    
    # plot collected results.
    fig, (axes_row1, axes_row2) = plt.subplots(2, len(error_types), figsize=(15, 5))
    for i, error_type in enumerate(error_types):
        for x_i, model_name in enumerate(model_names):
            model_name = model_names[x_i]
            if 'vgg16' in model_name:  
                output_layer = 'block5_pool'
                color = 'red'
            elif 'resnet50' in model_name:
                output_layer = 'avg_pool'
                color = 'green'
            elif 'vit' in model_name: 
                output_layer = 'layer_12'
                color = 'purple'

            mse = np.array(
                results_collector[error_type][model_name][output_layer]['mse'])
            ci_low = np.array(
                results_collector[error_type][model_name][output_layer]['ci'])[:, 0]
            ci_high = np.array(
                results_collector[error_type][model_name][output_layer]['ci'])[:, 1]
            
            # # TEMP
            # mse = _convert_mse_to_physical_unit(mse, error_type)
            # ci_low = _convert_mse_to_physical_unit(ci_low, error_type)
            # ci_high = _convert_mse_to_physical_unit(ci_high, error_type)

            if 'untrained' in model_name:
                mfc = 'white'
            else:
                mfc = None

            axes_row2[i].errorbar(
                x_i,
                mse,
                yerr=[mse-ci_low, ci_high-mse],
                label=model_name,
                marker='o',
                capsize=5,
                mfc=mfc,
                color=color
            )
            
            # set ylim with a bit of margin
            # axes_row2[i].set_ylim(
            #     np.min(mse-ci_low)-0.1, 
            #     np.max(ci_high-mse)+0.1
            # )
                            
        # baselines are the same for all models
        # so we only plot them once as plot
        baseline_predict_mid_mse = np.array(
            results_collector[error_type][model_name][output_layer]['baseline_predict_mid_mse'])
        
        # # TEMP
        # baseline_predict_mid_mse = _convert_mse_to_physical_unit(
        #     baseline_predict_mid_mse, error_type)
        
        axes_row1[i].plot(
            range(len(model_names)),
            baseline_predict_mid_mse.repeat(len(model_names)),
            label='baseline: center',
            color='cyan',
        )

        baseline_predict_random_mse = np.array(
            results_collector[error_type][model_name][output_layer]['baseline_predict_random_mse'])

        # # TEMP
        # baseline_predict_random_mse = _convert_mse_to_physical_unit(
        #     baseline_predict_random_mse, error_type)

        axes_row1[i].plot(
            range(len(model_names)),
            baseline_predict_random_mse.repeat(len(model_names)),
            label='baseline: random',
            color='blue',
        )
        # set ylim with a bit of margin 
        # two baselines can be different magnitude, 
        # so the upper bound is the max of the two with margin,
        # and the lower bound is the min of the two with margin.
        max_of_two_baseline = np.max(
            np.concatenate([baseline_predict_mid_mse, baseline_predict_random_mse])
        )
        min_of_two_baseline = np.min(
            np.concatenate([baseline_predict_mid_mse, baseline_predict_random_mse])
        )
        axes_row1[i].set_ylim(
            min_of_two_baseline-1, 
            max_of_two_baseline+1
        )

        # axes[i].set_ylabel('Decoding error (MSE)')
        if error_type == 'loc':  title = 'Location Decoding'
        elif error_type == 'rot': title = 'Direction Decoding'
        elif error_type == 'dist': title = 'Distance to Nearest Border Decoding'
        axes_row1[i].set_title(title)
        
        axes_row1[i].spines['bottom'].set_visible(False)
        axes_row1[i].spines['top'].set_visible(False)
        axes_row1[i].spines['right'].set_visible(False)
        axes_row2[i].spines['right'].set_visible(False)
        axes_row2[i].spines['top'].set_visible(False)
        axes_row1[i].set_xticks([])
        axes_row2[i].set_xticks(range(len(model_names)))
        axes_row2[i].set_xticklabels(model_names, rotation=90)

        # Add diagonal lines to connect the subplots
        d = 0.015  # How big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=axes_row1[i].transAxes, color='k', clip_on=False)
        axes_row1[i].plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal line
        kwargs.update(transform=axes_row2[i].transAxes)  # switch to the bottom subplot
        axes_row2[i].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal line

    axes_row1[2].legend(loc='upper right')
    fig.supylabel('Decoding error (MSE)')
    fig.tight_layout(rect=(0.025, 0, 1, 0.98))
    plt.savefig(f'figs/paper/decoding_across_models.png')
    plt.close()


def unit_chart_type_against_coef_each_model_across_layers():
    experiment = 'unit_chart_by_coef'
    envs = ['env28_r24']
    movement_mode = '2d'
    sampling_rate = 0.3
    random_seed = 42
    model_names = ['vgg16', 'resnet50', 'vit_b16']
    moving_trajectory = 'uniform'
    decoding_model_choice = {'name': 'ridge_regression', 'hparams': 1.0}
    feature_selection = 'l2'
    filterings = [
        {'filtering_order': 'top_n', 'n_units_filtering': None, 'p_units_filtering': 0.1},
        {'filtering_order': 'random_n', 'n_units_filtering': None, 'p_units_filtering': 0.1},
        {'filtering_order': 'mid_n', 'n_units_filtering': None, 'p_units_filtering': 0.1},
    ]

    unit_type_to_column_index_in_unit_chart = {
        'place_cell_1': {
            'num_clusters': 1,
        },
        'place_cell_2': {
            'max_value_in_clusters': 3,
        },
        'border_cell': {
            'borderness': 9,
        },
        'direction_cell': {
            'mean_vector_length': 10,
        },
    }

    for model_name in model_names:
        output_layers = data.load_model_layers(model_name)
        # each model has a figure, 
        # each row of a figure is a model layer
        # each column of a figure is a unit type (assoc. task)
        # e.g. axes[output_layer_i, metric_i]
        fig, axes = plt.subplots(
            len(output_layers), len(unit_type_to_column_index_in_unit_chart), 
            figsize=(5*len(unit_type_to_column_index_in_unit_chart), 5*len(output_layers))
        )

        envs_dict = data.load_envs_dict(model_name, envs)
        config_versions=list(envs_dict.keys())  
        # ken: suboptimal but each config_version is unique assoc. with a layer.
        for output_layer_i, config_version in enumerate(config_versions):
            output_layer = envs_dict[config_version]['output_layer']
            
            for unit_type_i, unit_type in enumerate(unit_type_to_column_index_in_unit_chart.keys()):
                metric = list(unit_type_to_column_index_in_unit_chart[unit_type].keys())[0]

                print(
                    f'[Plotting] {model_name} {output_layer} {unit_type} {metric} {config_version}'
                )

                if 'place_cell' in unit_type:
                    reference_experiment = 'loc_n_rot'
                    target = 'loc'
                elif unit_type == 'direction_cell':
                    reference_experiment = 'loc_n_rot'
                    target = 'rot'
                elif unit_type == 'border_cell':
                    reference_experiment = 'border_dist'
                    target = 'border_dist'
                
                # load presaved unit_chart info sorted by coef
                # using top, random, or mid filtering.
                # the unit_chart_info to be loaded needs to be
                # first computed by `inspect_model_units.py`
                config = utils.load_config(config_version)
                results_path = utils.load_results_path(
                    config=config,
                    experiment=experiment,
                    reference_experiment=reference_experiment,
                    feature_selection=feature_selection,
                    decoding_model_choice=decoding_model_choice,
                    sampling_rate=sampling_rate,
                    moving_trajectory=moving_trajectory,
                    random_seed=random_seed,
                )

                for filtering in filterings:
                    p_units_filtering = filtering['p_units_filtering']
                    filtering_order = filtering['filtering_order']
                    
                    unit_chart_info = np.load(
                        f'{results_path}/unit_chart_{target}_{filtering_order}_{p_units_filtering}.npy', 
                        allow_pickle=True
                    )

                    # plot distribution of metric and coef based on filtering_order
                    # `type_id` is the column index in unit_chart_info
                    type_id = unit_type_to_column_index_in_unit_chart[unit_type][metric]

                    # NOTE: not every type_id has the same data structure,
                    # e.g. for per unit place fields info, some columns might be ndarray 
                    # such as num_pixels_in_clusters where if the unit has multiple clusters,
                    # the entry for that unit for that column, will be an ndarray.
                    # So for plotting, we need to unpack these arrays.
                    unit_chart_info_unpacked = []
                    for x in unit_chart_info[:, type_id]:
                        if isinstance(x, np.ndarray):
                            unit_chart_info_unpacked.extend(x)
                        else:
                            unit_chart_info_unpacked.append(x)

                    if 'top' in filtering_order:
                        label = f'top {int(p_units_filtering*100):,}%'
                        color = 'green'
                    elif 'random' in filtering_order:
                        label = f'random {int(p_units_filtering*100):,}%'
                        color = 'purple'
                    elif 'mid' in filtering_order:
                        label = f'middle {int(p_units_filtering*100):,}%'
                        color = 'orange'

                    sns.kdeplot(
                        unit_chart_info_unpacked,
                        label=label,
                        color=color,
                        alpha=0.2,
                        fill=True,
                        ax=axes[output_layer_i, unit_type_i],
                    )

                if metric == 'num_clusters':
                    title = 'Place Tuning (1)'
                    x_label = 'Number of Place Fields'
                elif metric == 'max_value_in_clusters':
                    title = 'Place Tuning (2)'
                    x_label = 'Max Activity in Place Fields'
                elif metric == 'borderness':
                    title = 'Border Tuning'
                    x_label = 'Border Tuning Strength'
                elif metric == 'mean_vector_length':
                    title = 'Direction Tuning'
                    x_label = 'Directional Tuning Strength'
                axes[output_layer_i, unit_type_i].set_xlabel(x_label)
                axes[output_layer_i, unit_type_i].set_title(f'{title} ({output_layer})')
                axes[output_layer_i, unit_type_i].spines.right.set_visible(False)
                axes[output_layer_i, unit_type_i].spines.top.set_visible(False)
                axes[output_layer_i, 0].set_ylabel(f'Density')
            axes[output_layer_i, -1].legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'figs/paper/unit_chart_against_coef_{model_name}.png')
                

def lesion_by_coef_each_model_across_layers_and_lr():
    envs = ['env28_r24']
    env = envs[0]
    movement_mode = '2d'
    sampling_rate = 0.3
    random_seeds = [42]
    model_names = ['vgg16', 'resnet50', 'vit_b16']
    moving_trajectory = 'uniform'
    decoding_model_choice = {'name': 'ridge_regression', 'hparams': 1.0}
    decoding_model_name = decoding_model_choice['name']
    decoding_model_hparams = decoding_model_choice['hparams']
    base_feature_selection = 'l2'
    error_types = ['loc', 'rot', 'dist']
    tracked_metrics = ['mse', 'ci', 'baseline_predict_mid_mse', 'baseline_predict_random_mse']

    ranks = ['top', 'random']
    thr = 'thr'
    lesion_metric = 'coef'
    lesion_ratios = [0.0, 0.1, 0.3, 0.5, 0.7]

    results_collector = \
        defaultdict(                                # key - model_name
            lambda: defaultdict(                    # key - lesion rank
                lambda: defaultdict(                # key - error_type
                    lambda: defaultdict(            # key - output_layer
                        lambda: defaultdict(list)   # key - metric
                    )

                )
            )
        )
    
    for model_name in model_names:
        output_layers = data.load_model_layers(model_name)
        for rank in ranks:
            for error_type in error_types:
                if 'loc' in error_type:  
                    experiment = 'loc_n_rot'
                elif 'rot' in error_type:
                    experiment = 'loc_n_rot'
                elif 'dist' in error_type: 
                    experiment = 'border_dist'

                for output_layer in output_layers:
                    for lesion_ratio in lesion_ratios:
                        if lesion_ratio != 0:
                            if 'loc' in error_type:  
                                target = '_loc'
                            elif 'rot' in error_type:
                                target = '_rot'
                            elif 'dist' in error_type: 
                                target = '_borderdist'
                            feature_selection = f'{base_feature_selection}+lesion_{lesion_metric}_{thr}_{rank}_{lesion_ratio}{target}'
                        else:
                            feature_selection = base_feature_selection

                        # sampling rate would be the base dimension where 
                        # we accumulate results in a list to plot at once.
                        to_average_over_seeds = defaultdict(list)
                        for random_seed in random_seeds:
                            results_path = \
                                f'results/{env}/{movement_mode}/{moving_trajectory}/'\
                                f'{model_name}/{experiment}/{feature_selection}/'\
                                f'{decoding_model_name}_{decoding_model_hparams}/'\
                                f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
                            results = np.load(f'{results_path}/res.npy', allow_pickle=True).item()[error_type]
                            for metric in tracked_metrics:
                                to_average_over_seeds[metric].append(results[metric])
                            
                        # per metric per output layer 
                        # across sampling rates averaged over seeds
                        for metric in tracked_metrics:
                            # a special case is when metric=='ci' where 
                            # ..res[metric] is a list of 2 elements
                            # so we need to average wrt each element across seeds
                            # and save them back as 2 elements for later plotting.
                            if metric == 'ci':
                                ci_low_avg = np.mean(
                                    [ci[0] for ci in to_average_over_seeds[metric]])
                                ci_high_avg = np.mean(
                                    [ci[1] for ci in to_average_over_seeds[metric]])
                                avg_res = [ci_low_avg, ci_high_avg]
                            else:
                                avg_res = np.mean(to_average_over_seeds[metric])
                            results_collector[model_name][rank][error_type][output_layer][metric].append(avg_res)
    
    # plot collected results.
    # each subplot has coord axes[rank_i, error_type_i]
    for model_name in model_names:
        output_layers = data.load_model_layers(model_name)
        
        fig, axes = plt.subplots(len(ranks), len(error_types), figsize=(15, 10))
        for rank_i, rank in enumerate(ranks):
            for error_type_i, error_type in enumerate(error_types):
                for output_layer in output_layers:
                    for metric in tracked_metrics:
                        if metric == 'ci':
                            ci_low = np.array(
                                results_collector[model_name][rank][error_type][output_layer]['ci'])[:, 0]
                            ci_high = np.array(
                                results_collector[model_name][rank][error_type][output_layer]['ci'])[:, 1]
                            axes[rank_i, error_type_i].fill_between(
                                lesion_ratios,
                                ci_low,
                                ci_high,
                                alpha=0.3,
                                color='grey',
                            )
                        else:
                            if 'baseline' in metric:
                                # no need to label baseline for each layer
                                # we only going to label baseline when we plot
                                # the last layer.
                                if output_layer == output_layers[-1]:
                                    if 'mid' in metric:
                                        label = 'baseline: center'
                                    else:
                                        label = 'baseline: random'
                                else:
                                    label = None  
                                if 'mid' in metric: 
                                    color = 'cyan'
                                else: 
                                    color = 'blue'
                            else:
                                # for non-baseline layer performance,
                                # we label each layer and use layer-specific color.
                                label = output_layer
                                if "predictions" in label: label = "logits"
                                color = data.load_envs_dict(model_name, envs)[
                                    f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                            
                            # either baseline or non-baseline layer performance,
                            # we always plot them.
                            axes[rank_i, error_type_i].plot(
                                lesion_ratios,
                                results_collector[model_name][rank][error_type][output_layer][metric],
                                label=label,
                                color=color,
                                marker='o',
                            )
                axes[rank_i, error_type_i].set_xlabel('Lesion ratio')
                axes[rank_i, error_type_i].set_ylabel('Decoding error (MSE)')
                axes[rank_i, error_type_i].set_xticks(lesion_ratios)
                axes[rank_i, error_type_i].set_xticklabels(lesion_ratios)
                if rank == 'top':
                    if error_type == 'loc':  title = 'Top Coefficient Lesion\n(Location Decoding)'
                    elif error_type == 'rot': title = 'Top Coefficient Lesion\n(Direction Decoding)'
                    elif error_type == 'dist': title = 'Top Coefficient Lesion\n(Distance to Nearest Border Decoding)'
                elif rank == 'random':
                    if error_type == 'loc':  title = 'Random Coefficient Lesion\n(Location Decoding)'
                    elif error_type == 'rot': title = 'Random Coefficient Lesion\n(Direction Decoding)'
                    elif error_type == 'dist': title = 'Random Coefficient Lesion\n(Distance to Nearest Border Decoding)'
                axes[rank_i, error_type_i].set_title(title)
                axes[rank_i, error_type_i].spines.right.set_visible(False)
                axes[rank_i, error_type_i].spines.top.set_visible(False)
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.savefig(f'figs/paper/lesion_by_coef_{model_name}.png')
        plt.close()


def lesion_by_unit_chart_each_model_across_layers_and_lr():
    """
    maxvalueinclusters (loc), numclusters (loc), directioness (rot), borderness (border)
    """
    envs = ['env28_r24']
    env = envs[0]
    movement_mode = '2d'
    sampling_rate = 0.3
    random_seeds = [42]
    model_names = ['vgg16', 'resnet50', 'vit_b16']
    moving_trajectory = 'uniform'
    decoding_model_choice = {'name': 'ridge_regression', 'hparams': 1.0}
    decoding_model_name = decoding_model_choice['name']
    decoding_model_hparams = decoding_model_choice['hparams']
    base_feature_selection = 'l2'
    unit_chart_types = ['maxvalueinclusters', 'numclusters', 'directioness', 'borderness']
    tracked_metrics = ['mse', 'ci', 'baseline_predict_mid_mse', 'baseline_predict_random_mse']

    ranks = ['top', 'random']
    thr = '0'
    lesion_ratios = [0.0, 0.1, 0.3, 0.5, 0.7]

    results_collector = \
        defaultdict(                                # key - model_name
            lambda: defaultdict(                    # key - lesion rank
                lambda: defaultdict(                # key - unit_chart_type (assoc. error_type)
                    lambda: defaultdict(            # key - output_layer
                        lambda: defaultdict(list)   # key - metric
                    )

                )
            )
        )
    
    for model_name in model_names:
        output_layers = data.load_model_layers(model_name)
        for rank in ranks:
            for unit_chart_type in unit_chart_types:
                lesion_metric = unit_chart_type
                if 'maxvalueinclusters' in unit_chart_type or 'numclusters' in unit_chart_type:  
                    experiment = 'loc_n_rot'
                    error_type = 'loc'
                elif 'directioness' in unit_chart_type:
                    experiment = 'loc_n_rot'
                    error_type = 'rot'
                elif 'borderness' in unit_chart_type:
                    experiment = 'border_dist'
                    error_type = 'dist'

                for output_layer in output_layers:
                    for lesion_ratio in lesion_ratios:
                        if lesion_ratio != 0:
                            feature_selection = f'{base_feature_selection}+lesion_{lesion_metric}_{thr}_{rank}_{lesion_ratio}'
                        else:
                            feature_selection = base_feature_selection

                        # sampling rate would be the base dimension where 
                        # we accumulate results in a list to plot at once.
                        to_average_over_seeds = defaultdict(list)
                        for random_seed in random_seeds:
                            results_path = \
                                f'results/{env}/{movement_mode}/{moving_trajectory}/'\
                                f'{model_name}/{experiment}/{feature_selection}/'\
                                f'{decoding_model_name}_{decoding_model_hparams}/'\
                                f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
                            results = np.load(f'{results_path}/res.npy', allow_pickle=True).item()[error_type]
                            for metric in tracked_metrics:
                                to_average_over_seeds[metric].append(results[metric])
                            
                        # per metric per output layer 
                        # across sampling rates averaged over seeds
                        for metric in tracked_metrics:
                            # a special case is when metric=='ci' where 
                            # ..res[metric] is a list of 2 elements
                            # so we need to average wrt each element across seeds
                            # and save them back as 2 elements for later plotting.
                            if metric == 'ci':
                                ci_low_avg = np.mean(
                                    [ci[0] for ci in to_average_over_seeds[metric]])
                                ci_high_avg = np.mean(
                                    [ci[1] for ci in to_average_over_seeds[metric]])
                                avg_res = [ci_low_avg, ci_high_avg]
                            else:
                                avg_res = np.mean(to_average_over_seeds[metric])
                            results_collector[model_name][rank][unit_chart_type][output_layer][metric].append(avg_res)
    
    # plot collected results.
    # each subplot has coord axes[rank_i, unit_chart_type]
    for model_name in model_names:
        output_layers = data.load_model_layers(model_name)
        
        fig, axes = plt.subplots(len(ranks), len(unit_chart_types), figsize=(15, 10))
        for rank_i, rank in enumerate(ranks):
            for unit_chart_type_i, unit_chart_type in enumerate(unit_chart_types):
                for output_layer in output_layers:
                    for metric in tracked_metrics:
                        if metric == 'ci':
                            ci_low = np.array(
                                results_collector[model_name][rank][unit_chart_type][output_layer]['ci'])[:, 0]
                            ci_high = np.array(
                                results_collector[model_name][rank][unit_chart_type][output_layer]['ci'])[:, 1]
                            axes[rank_i, unit_chart_type_i].fill_between(
                                lesion_ratios,
                                ci_low,
                                ci_high,
                                alpha=0.3,
                                color='grey',
                            )
                        else:
                            if 'baseline' in metric:
                                # no need to label baseline for each layer
                                # we only going to label baseline when we plot
                                # the last layer.
                                if output_layer == output_layers[-1]:
                                    if 'mid' in metric:
                                        label = 'baseline: center'
                                    else:
                                        label = 'baseline: random'
                                else:
                                    label = None  
                                if 'mid' in metric: 
                                    color = 'cyan'
                                else: 
                                    color = 'blue'
                            else:
                                # for non-baseline layer performance,
                                # we label each layer and use layer-specific color.
                                label = output_layer
                                if "predictions" in label: label = "logits"
                                color = data.load_envs_dict(model_name, envs)[
                                    f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                            
                            # either baseline or non-baseline layer performance,
                            # we always plot them.
                            axes[rank_i, unit_chart_type_i].plot(
                                lesion_ratios,
                                results_collector[model_name][rank][unit_chart_type][output_layer][metric],
                                label=label,
                                color=color,
                                marker='o',
                            )
                axes[rank_i, unit_chart_type_i].set_xlabel('Lesion ratio')
                axes[rank_i, unit_chart_type_i].set_ylabel('Decoding error (MSE)')
                axes[rank_i, unit_chart_type_i].set_xticks(lesion_ratios)
                axes[rank_i, unit_chart_type_i].set_xticklabels(lesion_ratios)
                if rank == 'top':
                    if unit_chart_type == 'maxvalueinclusters':
                        title = 'Top Place Field Activity Lesion\n(Location Decoding)'
                    elif unit_chart_type == 'numclusters':
                        title = 'Top Place Field Quantity Lesion\n(Location Decoding)'
                    elif unit_chart_type == 'directioness':
                        title = 'Top Directional Tuning Lesion\n(Direction Decoding)'
                    elif unit_chart_type == 'borderness':
                        title = 'Top Border Tuning Lesion\n(Distance to Nearest Border Decoding)'
                elif rank == 'random':
                    if unit_chart_type == 'maxvalueinclusters':
                        title = 'Random Place Field Activity Lesion\n(Location Decoding)'
                    elif unit_chart_type == 'numclusters':
                        title = 'Random Place Field Quantity Lesion\n(Location Decoding)'
                    elif unit_chart_type == 'directioness':
                        title = 'Random Directional Tuning Lesion\n(Direction Decoding)'
                    elif unit_chart_type == 'borderness':
                        title = 'Random Border Tuning Lesion\n(Distance to Nearest Border Decoding)'
                
                axes[rank_i, unit_chart_type_i].set_title(title)
                axes[rank_i, unit_chart_type_i].spines.right.set_visible(False)
                axes[rank_i, unit_chart_type_i].spines.top.set_visible(False)
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.savefig(f'figs/paper/lesion_by_unit_chart_{model_name}.png')
        plt.close()


def unit_visualization_by_type():
    config_version = 'env28_r24_2d_vgg16_fc2'
    experiment = 'unit_chart'
    moving_trajectory = 'uniform'
    config = utils.load_config(config_version)

    # load model outputs
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{TF_NUM_INTRAOP_THREADS}"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    
    # load model outputs
    if config['model_name'] == 'none':
        model = None
        preprocess_func = None
    else:
        model, preprocess_func = models.load_model(
            config['model_name'], config['output_layer'])

    preprocessed_data = data.load_preprocessed_data(
        config=config,
        data_path=\
            f"data/unity/"\
            f"{config['unity_env']}/"\
            f"{config['movement_mode']}", 
        movement_mode=config['movement_mode'],
        env_x_min=config['env_x_min'],
        env_x_max=config['env_x_max'],
        env_y_min=config['env_y_min'],
        env_y_max=config['env_y_max'],
        multiplier=config['multiplier'],
        n_rotations=config['n_rotations'],
        preprocess_func=preprocess_func,
    )    
    
    # (n_locations*n_rotations, n_features)
    model_reps = data.load_full_dataset_model_reps(
        config, model, preprocessed_data
    )

    # reshape to (n_locations, n_rotations, n_features)
    model_reps = model_reps.reshape(
        (model_reps.shape[0] // config['n_rotations'],  # n_locations
        config['n_rotations'],                          # n_rotations
        model_reps.shape[1])                            # all units
    )
    model_reps_summed = np.sum(model_reps, axis=1, keepdims=True)

    # load unit chart info
    results_path = utils.load_results_path(
        config=config,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    unit_chart_info = np.load(f'{results_path}/unit_chart.npy', allow_pickle=True)
    per_rotation_vector_length = unit_chart_info[:, 11]
    movement_mode=config['movement_mode']
    env_x_min=config['env_x_min']
    env_x_max=config['env_x_max']
    env_y_min=config['env_y_min']
    env_y_max=config['env_y_max']
    multiplier=config['multiplier']

    # plottings
    for unit_type in ['place_cell_1_field', 'place_cell_n_fields', 'border_cell']:
        if unit_type == 'place_cell_1_field':
            # single field place cell
            selected_n_indices = [2631, 8, 2803, 1654, 475, 4055]
        if unit_type == 'place_cell_n_fields':
            # multiple fields place cell
            selected_n_indices = [1472, 3507, 1115, 2076, 679, 3226]
        if unit_type == 'border_cell':
            # border cells
            selected_n_indices = [3866, 2404, 1476, 2433, 3846, 2949]

        fig = plt.figure(figsize=(10, 3))
        # ref - https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot
        gs = gridspec.GridSpec(
            nrows=2, 
            ncols=len(selected_n_indices),
            width_ratios=[1]*len(selected_n_indices),
            wspace=0.1, 
            hspace=0.3,
            top=0.9,
            bottom=0.15,
            left=0.05,
            right=0.95,
        )

        # one column is a unit, 
        # first row is ratemap, second row is polar plot
        for col_index, unit_index in enumerate(selected_n_indices):
            for rotation in range(model_reps_summed.shape[1]):
                if movement_mode == '2d':
                    # reshape to (n_locations, n_rotations, n_features)
                    heatmap = model_reps_summed[:, rotation, unit_index].reshape(
                        (env_x_max*multiplier-env_x_min*multiplier+1, 
                        env_y_max*multiplier-env_y_min*multiplier+1) )
                    # rotate heatmap to match Unity coordinate system
                    # ref: tests/testReshape_forHeatMap.py
                    heatmap = np.rot90(heatmap, k=1, axes=(0, 1))

                    # --- subplot1: plot heatmap for the selected units ---
                    ax = fig.add_subplot(gs[0, col_index])
                    ax.imshow(heatmap, cmap='jet', interpolation='nearest')
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # --- subplot2: plot polar plot for the selected units ---
                    ax = fig.add_subplot(gs[1, col_index], projection='polar')
                    # we want to have a closed loop
                    theta = np.linspace(0, 2*np.pi, model_reps.shape[1], endpoint=False)
                    x = theta.tolist() + [theta[0]]
                    y = per_rotation_vector_length[unit_index] + [per_rotation_vector_length[unit_index][0]]
                    ax.plot(x, y)
                    ax.set_theta_zero_location("N")
                    ax.set_theta_direction(-1)
                    ax.set_rticks([])
                    if col_index == 0:
                        ax.set_thetagrids([0, 90, 180, 270], labels=['0', '90', '180', '270'])
                    else:
                        ax.set_thetagrids([0, 90, 180, 270], labels=['', '', '', ''])
        
        plt.savefig(f'figs/paper/unit_visualization_by_type_{unit_type}.png')
        plt.close()
      

if __name__ == '__main__':
    TF_NUM_INTRAOP_THREADS = 10
    # decoding_each_model_across_layers_and_sr()
    # decoding_all_models_one_layer_one_sr()
    # lesion_by_coef_each_model_across_layers_and_lr()
    # lesion_by_unit_chart_each_model_across_layers_and_lr()
    # unit_chart_type_against_coef_each_model_across_layers()
    unit_visualization_by_type()

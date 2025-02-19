import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

import data
import utils
import models
import unit_metric_computers as umc

plt.rcParams.update({'font.size': 22, })

output_layers_2_levels = {
    "vgg16": {
        "block2_pool": "Early (block2_pool)",
        "block4_pool": "Mid (block4_pool)",
        "block5_pool": "Late (block5_pool)",
        "fc2": "Penultimate (fc2)",
    },
    "vgg16_untrained": {
        "block2_pool": "Early (block2_pool)",
        "block4_pool": "Mid (block4_pool)",
        "block5_pool": "Late (block5_pool)",
        "fc2": "Penultimate (fc2)",
    },
    "resnet50": {
        "conv2_block3_out": "Early (conv2_block3_out)",
        "conv4_block6_out": "Mid (conv4_block6_out)",
        "conv5_block2_out": "Late (conv5_block2_out)",
        "avg_pool": "Penultimate (avg_pool)",
    },
    "resnet50_untrained": {
        "conv2_block3_out": "Early (conv2_block3_out)",
        "conv4_block6_out": "Mid (conv4_block6_out)",
        "conv5_block2_out": "Late (conv5_block2_out)",
        "avg_pool": "Penultimate (avg_pool)",
    },
    "vit_b16": {
        "layer_3": "Early (layer_3)",
        "layer_6": "Mid (layer_6)",
        "layer_9": "Late (layer_9)",
        "layer_12": "Penultimate (layer_12)",
    },
    "vit_b16_untrained": {
        "layer_3": "Early (layer_3)",
        "layer_6": "Mid (layer_6)",
        "layer_9": "Late (layer_9)",
        "layer_12": "Penultimate (layer_12)",
    },
}

model_names_2_pretty_names = {
    "vgg16": "VGG-16",
    "vgg16_untrained": "VGG-16 (untrained)",
    "resnet50": "ResNet-50",
    "resnet50_untrained": "ResNet-50 (untrained)",
    "vit_b16": "ViT-B/16",
    "vit_b16_untrained": "ViT-B/16 (untrained)",
}


def _convert_mse_to_physical_unit(mse, error_type, normalized=True):
    """
    Convert MSE error back to physical sense (Unity units or degree).

    For location and distance error, we first take the square root of MSE,
    which results in 'how far off the prediction is from truth' in terms of
    the relative coordinate system where the targets are defined (from -5 to 5
    on each axis). We then convert this error to Unity units. In Unity, the length
    of the moving area is 2, which means a wall with length=10 in the relative
    coordinate system is 2 in Unity. Therefore, the square root of MSE needs
    further scaled down by 5 to map error back to Unity units.

    For rotation error, we map the error back to degree by first 
    taking the square root of MSE, which gives 'how many intervals'
    is the prediction off, and since we have 24 intervals out of 360 degrees,
    we can map the error back to degree by multiplying the square root of MSE
    by 360/24.

    If `normalized=True`, we further normalize the error by dividing it by the
    maximum possible error, which is 2 for location and distance error, and 180
    for rotation error.

    `normalized` can be set by `normalize_error` in the global scope.
    """
    coordinate_system_to_unity_scale = 2 / 10
    if error_type == 'loc' or error_type == 'dist':
        if normalized:
            return np.sqrt(mse) * coordinate_system_to_unity_scale / 2
        return np.sqrt(mse) * coordinate_system_to_unity_scale
    
    elif error_type == 'rot':
        if normalized:
            return np.sqrt(mse) * 360/24 / 180
        return np.sqrt(mse) * 360/24

    else:
        raise ValueError(f"Unknown error type: {error_type}")


def decoding_each_model_across_layers_and_sr():
    envs = ['env28_r24']
    env = envs[0]
    movement_mode = '2d'
    sampling_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    random_seeds = [42]
    model_names = [
        'vgg16', 
        'vgg16_untrained', 
        'resnet50', 'resnet50_untrained',
        'vit_b16', 'vit_b16_untrained'
    ]
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
        fig, axes = plt.subplots(1, len(error_types), figsize=(18, 5))
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
                        
                        # TEMP
                        ci_low = _convert_mse_to_physical_unit(ci_low, error_type, normalized=normalize_error)
                        ci_high = _convert_mse_to_physical_unit(ci_high, error_type, normalized=normalize_error)

                        axes[i].fill_between(
                            sampling_rates,
                            ci_low,
                            ci_high,
                            alpha=1,
                            color='#DADADA',
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
                                color = '#8B9FA5'
                            else: 
                                color = '#9ABA79'
                        else:
                            # for non-baseline layer performance,
                            # we label each layer and use layer-specific color.
                            label = output_layers_2_levels[model_name][output_layer]
                            if "predictions" in label: label = "logits"
                            color = data.load_envs_dict(model_name, envs)[
                                f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                        
                        # either baseline or non-baseline layer performance,
                        # we always plot them.
                        axes[i].plot(
                            sampling_rates,
                            # results_collector[error_type][output_layer][metric],
                            # TEMP
                            _convert_mse_to_physical_unit(
                                np.array(results_collector[error_type][output_layer][metric]),
                                error_type,
                                normalized=normalize_error,
                            ),
                            label=label,
                            color=color,
                            marker='o',
                        )
            axes[i].set_xlabel('Sampling rate')
            axes[i].set_xticks(sampling_rates)
            axes[i].set_xticklabels(sampling_rates)
            if normalize_error is True:
                ax_label = 'Normalized Error'
            elif normalize_error is False:
                ax_label = 'Error (physical)'
            else:
                raise ValueError(f"Unknown value for normalize_error: {normalize_error}")
            if error_type == 'loc':  
                title = 'Location Decoding'
                axes[i].set_ylabel(ax_label)
            elif error_type == 'rot': 
                title = 'Direction Decoding'
                axes[i].set_ylabel(ax_label)
            elif error_type == 'dist': 
                title = 'Nearest Border Decoding'
                axes [i].set_ylabel(ax_label)
            axes[i].set_title(title)
            axes[i].spines.right.set_visible(False)
            axes[i].spines.top.set_visible(False)
        
        plt.subplots_adjust(right=0.8)
        plt.legend(fontsize=22, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        if normalize_error is True:
            plt.savefig(f'figs/paper/decoding_{model_name}.png')
            plt.savefig(f'figs/paper/decoding_{model_name}.svg')
        elif normalize_error is False:
            plt.savefig(f'figs/paper/decoding_{model_name}_unnorm.png')
            plt.savefig(f'figs/paper/decoding_{model_name}_unnorm.svg')
        else:
            raise ValueError(f"Unknown value for normalize_error: {normalize_error}")
        plt.close()


def TEMP__decoding_each_model_across_layers_and_sr_V2():
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
        # fig, axes = plt.subplots(1, len(error_types), figsize=(15, 5))
        fig, (axes_row1, axes_row2) = plt.subplots(2, len(error_types), figsize=(12, 8))
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
                        
                        # TEMP
                        ci_low = _convert_mse_to_physical_unit(ci_low, error_type, normalized=normalize_error)
                        ci_high = _convert_mse_to_physical_unit(ci_high, error_type, normalized=normalize_error)

                        axes_row2[i].fill_between(
                            sampling_rates,
                            ci_low,
                            ci_high,
                            alpha=1,
                            color='#DADADA',
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
                                color = '#8B9FA5'
                            else: 
                                color = '#9ABA79'
                            ax_to_plot = axes_row1[i]
                        else:
                            # for non-baseline layer performance,
                            # we label each layer and use layer-specific color.
                            label = output_layer
                            if "predictions" in label: label = "logits"
                            color = data.load_envs_dict(model_name, envs)[
                                f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                            ax_to_plot = axes_row2[i]
                        
                        # either baseline or non-baseline layer performance,
                        # we always plot them.
                        ax_to_plot.plot(
                            sampling_rates,
                            # results_collector[error_type][output_layer][metric],
                            # TEMP
                            _convert_mse_to_physical_unit(
                                np.array(results_collector[error_type][output_layer][metric]),
                                error_type,
                                normalized=normalize_error,
                            ),
                            label=label,
                            color=color,
                            marker='o',

                        )
            if error_type == 'loc':  
                title = 'Location Decoding'
                axes_row1[i].set_ylabel("Normalized Error")
                axes_row2[i].set_ylabel("Normalized Error")
            elif error_type == 'rot': 
                title = 'Direction Decoding'
                axes_row1[i].set_ylabel("Normalized Error")
                axes_row2[i].set_ylabel("Normalized Error")
            elif error_type == 'dist': 
                title = 'Nearest Border Decoding'
                axes_row1[i].set_ylabel("Normalized Error")
                axes_row2[i].set_ylabel("Normalized Error")
            axes_row2[i].set_xlabel('Sampling rate')
            axes_row2[i].set_xticks(sampling_rates)
            axes_row2[i].set_xticklabels(sampling_rates)
            axes_row1[i].spines['bottom'].set_visible(False)
            axes_row1[i].spines['top'].set_visible(False)
            axes_row1[i].spines['right'].set_visible(False)
            axes_row2[i].spines['right'].set_visible(False)
            axes_row2[i].spines['top'].set_visible(False)
            axes_row1[i].set_xticks([])
            axes_row1[i].set_title(title)
            if i == 0: subplot_label = 'A'
            elif i == 1: subplot_label = 'B'
            elif i == 2: subplot_label = 'C'
            axes_row1[i].text(-0.2, 1.1, subplot_label, fontsize=14, fontweight='bold',
            transform=axes_row1[i].transAxes, va='top', ha='left')

            # Add diagonal lines to connect the subplots
            d = 0.015  # How big to make the diagonal lines in axes coordinates
            kwargs = dict(transform=axes_row1[i].transAxes, color='k', clip_on=False)
            axes_row1[i].plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal line
            kwargs.update(transform=axes_row2[i].transAxes)  # switch to the bottom subplot
            axes_row2[i].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal line

        axes_row1[-1].legend(loc='upper right')
        axes_row2[-1].legend(loc='upper right')
        # fig.supylabel('Virtual environment units')
        fig.tight_layout(rect=[0, 0.0, 1, 0.99])
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
    model_names = [
        'vgg16', 
        'vgg16_untrained', 
        'resnet50', 'resnet50_untrained',
        'vit_b16', 'vit_b16_untrained'
    ]
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
    fig, (axes_row1, axes_row2) = plt.subplots(2, len(error_types), figsize=(18, 5))
    for i, error_type in enumerate(error_types):
        for x_i, model_name in enumerate(model_names):
            model_name = model_names[x_i]
            if 'vgg16' in model_name:  
                output_layer = 'block5_pool'
                fillcolor = '#33539E'
            elif 'resnet50' in model_name:
                output_layer = 'avg_pool'
                fillcolor = '#BFBBDA'
            elif 'vit' in model_name: 
                output_layer = 'layer_12'
                fillcolor = '#A5678E'

            mse = np.array(
                results_collector[error_type][model_name][output_layer]['mse'])
            ci_low = np.array(
                results_collector[error_type][model_name][output_layer]['ci'])[:, 0]
            ci_high = np.array(
                results_collector[error_type][model_name][output_layer]['ci'])[:, 1]
            
            # TEMP
            mse = _convert_mse_to_physical_unit(mse, error_type, normalized=normalize_error)
            ci_low = _convert_mse_to_physical_unit(ci_low, error_type, normalized=normalize_error)
            ci_high = _convert_mse_to_physical_unit(ci_high, error_type, normalized=normalize_error)

            if 'untrained' in model_name:
                edgecolor = fillcolor
                fillcolor = 'white'
            else:
                edgecolor = None

            # plot barplot with error bars
            axes_row2[i].bar(
                x_i,
                mse,
                yerr=[mse-ci_low, ci_high-mse],
                label=model_name,
                color=fillcolor,
                edgecolor=edgecolor,
                capsize=5,
            )
                              
        # baselines are the same for all models
        # so we only plot them once as plot
        baseline_predict_mid_mse = np.array(
            results_collector[error_type][model_name][output_layer]['baseline_predict_mid_mse'])
        
        # TEMP
        baseline_predict_mid_mse = _convert_mse_to_physical_unit(
            baseline_predict_mid_mse, error_type, normalized=normalize_error
        )
        
        axes_row1[i].plot(
            range(len(model_names)),
            baseline_predict_mid_mse.repeat(len(model_names)),
            label='baseline: center',
            color='#8B9FA5',
        )

        baseline_predict_random_mse = np.array(
            results_collector[error_type][model_name][output_layer]['baseline_predict_random_mse'])

        # TEMP
        baseline_predict_random_mse = _convert_mse_to_physical_unit(
            baseline_predict_random_mse, error_type, normalized=normalize_error
        )

        axes_row1[i].plot(
            range(len(model_names)),
            baseline_predict_random_mse.repeat(len(model_names)),
            label='baseline: random',
            color='#9ABA79',
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

        if normalize_error is True:
            ax_label = 'Normalized Error'
        elif normalize_error is False:
            ax_label = 'Error (physical)'
        else:
            raise ValueError(f"Unknown value for normalize_error: {normalize_error}")

        if error_type == 'loc':  
            title = 'Location Decoding'
            axes_row2[i].set_ylabel(' '*14+ax_label)
            axes_row1[i].set_ylim(0.2, 1.05)
        elif error_type == 'rot': 
            title = 'Direction Decoding'
            axes_row2[i].set_ylabel(' '*14+ax_label, labelpad=10)
        elif error_type == 'dist': 
            title = 'Nearest Border Decoding'
            axes_row2[i].set_ylabel(' '*14+ax_label)
            axes_row1[i].set_ylim(0.2, 1.05)
        
        axes_row1[i].set_title(title)
        axes_row1[i].spines['bottom'].set_visible(False)
        axes_row1[i].spines['top'].set_visible(False)
        axes_row1[i].spines['right'].set_visible(False)
        axes_row2[i].spines['right'].set_visible(False)
        axes_row2[i].spines['top'].set_visible(False)
        axes_row1[i].set_xticks([])
        axes_row2[i].set_xticks(range(len(model_names)))

        # pretty model names
        pretty_model_names = []
        for model_name in model_names:
            if model_name == 'vgg16': pretty_model_names.append('VGG-16')
            if model_name == 'vgg16_untrained': pretty_model_names.append('VGG-16\n(untrained)')
            if model_name == 'resnet50': pretty_model_names.append('ResNet-50')
            if model_name == 'resnet50_untrained': pretty_model_names.append('ResNet-50\n(untrained)')
            if model_name == 'vit_b16': pretty_model_names.append('ViT-B/16')
            if model_name == 'vit_b16_untrained': pretty_model_names.append('ViT-B/16\n(untrained)')
        # if there is untrained in model_name, make the label font color grey
        # otherwise use the default black.
        axes_row2[i].set_xticklabels(pretty_model_names, rotation=90)
        for x_i, model_name in enumerate(model_names):
            if 'untrained' in model_name:
                axes_row2[i].get_xticklabels()[x_i].set_color('grey')

        # Add diagonal lines to connect the subplots
        d = 0.015  # How big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=axes_row1[i].transAxes, color='k', clip_on=False)
        axes_row1[i].plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal line
        kwargs.update(transform=axes_row2[i].transAxes)  # switch to the bottom subplot
        axes_row2[i].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal line

    fig.tight_layout(rect=(0.02, 0, 1, 0.99))
    if normalize_error is True:
        plt.savefig(f'figs/paper/decoding_across_models.svg')
        plt.savefig(f'figs/paper/decoding_across_models.pdf')
    elif normalize_error is False:
        plt.savefig(f'figs/paper/decoding_across_models_unnorm.svg')
        plt.savefig(f'figs/paper/decoding_across_models_unnorm.pdf')
    else:
        raise ValueError(f"Unknown value for normalize_error: {normalize_error}")
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
    lesion_ratios = [0, 0.1, 0.3, 0.5, 0.7]

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
        
        fig, axes = plt.subplots(len(ranks), len(error_types), figsize=(18, 11))
        for rank_i, rank in enumerate(ranks):
            for error_type_i, error_type in enumerate(error_types):
                for output_layer in output_layers:
                    for metric in tracked_metrics:
                        if metric == 'ci':
                            ci_low = np.array(
                                results_collector[model_name][rank][error_type][output_layer]['ci'])[:, 0]
                            ci_high = np.array(
                                results_collector[model_name][rank][error_type][output_layer]['ci'])[:, 1]

                            # TEMP
                            ci_low = _convert_mse_to_physical_unit(ci_low, error_type, normalized=normalize_error)
                            ci_high = _convert_mse_to_physical_unit(ci_high, error_type, normalized=normalize_error)
                            
                            axes[rank_i, error_type_i].fill_between(
                                lesion_ratios,
                                ci_low,
                                ci_high,
                                alpha=1,
                                color='#DADADA',
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
                                    color = '#8B9FA5'
                                else: 
                                    color = '#9ABA79'
                            else:
                                # for non-baseline layer performance,
                                # we label each layer and use layer-specific color.
                                label = output_layers_2_levels[model_name][output_layer]
                                if "predictions" in label: label = "logits"
                                color = data.load_envs_dict(model_name, envs)[
                                    f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                            
                            # either baseline or non-baseline layer performance,
                            # we always plot them.
                            axes[rank_i, error_type_i].plot(
                                lesion_ratios,
                                # results_collector[model_name][rank][error_type][output_layer][metric],
                                # TEMP
                                _convert_mse_to_physical_unit(
                                    np.array(
                                        results_collector[model_name][rank][error_type][output_layer][metric]
                                    ), 
                                    error_type,
                                    normalized=normalize_error,
                                ),
                                label=label,
                                color=color,
                                marker='o',
                            )
                axes[rank_i, error_type_i].set_xlabel('Exclusion ratio')
                if normalize_error is True:
                    ax_label = 'Normalized Error'
                elif normalize_error is False:
                    ax_label = 'Error (physical)'
                else:
                    raise ValueError(f"Unknown value for normalize_error: {normalize_error}")
                if error_type in ['loc', 'dist']:
                    axes[rank_i, error_type_i].set_ylabel(ax_label)
                elif error_type == 'rot':
                    axes[rank_i, error_type_i].set_ylabel(ax_label, labelpad=10)
                axes[rank_i, error_type_i].set_xticks(lesion_ratios)
                axes[rank_i, error_type_i].set_xticklabels(lesion_ratios)
                if rank == 'top':
                    if error_type == 'loc':  title = 'Top Coef.\n(Location)'
                    elif error_type == 'rot': title = 'Top Coef.\n(Direction)'
                    elif error_type == 'dist': title = 'Top Coef.\n(Nearest Border)'
                elif rank == 'random':
                    if error_type == 'loc':  title = 'Random Coef.\n(Location)'
                    elif error_type == 'rot': title = 'Random Coef.\n(Direction)'
                    elif error_type == 'dist': title = 'Random Coef.\n(Nearest Border)'
                axes[rank_i, error_type_i].set_title(title)
                axes[rank_i, error_type_i].spines.right.set_visible(False)
                axes[rank_i, error_type_i].spines.top.set_visible(False)

        axes[1, 0].legend(
            fontsize=22, loc='upper left', bbox_to_anchor=(0.1, -0.2), ncol=3
        )
        # create more space at the bottom 
        # so that the legend is not cut off.
        plt.subplots_adjust(bottom=0.2, hspace=.6, wspace=.8)
        # create more space between two rows of subplots
        plt.tight_layout()
        if normalize_error is True:
            plt.savefig(f'figs/paper/lesion_by_coef_{model_name}.png')
            plt.savefig(f'figs/paper/lesion_by_coef_{model_name}.svg')
        elif normalize_error is False:
            plt.savefig(f'figs/paper/lesion_by_coef_{model_name}_unnorm.png')
            plt.savefig(f'figs/paper/lesion_by_coef_{model_name}_unnorm.svg')
        else:
            raise ValueError(f"Unknown value for normalize_error: {normalize_error}")
        plt.close()


def TEMP__lesion_by_coef_each_model_across_layers_and_lr_V2():
    envs = ['env28_r24']
    env = envs[0]
    movement_mode = '2d'
    sampling_rate = 0.3
    random_seeds = [42]
    # model_names = ['vgg16', 'resnet50', 'vit_b16']
    model_names = ['vgg16']
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
    lesion_ratios = [0, 0.1, 0.3, 0.5, 0.7]

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
        
        for rank_i, rank in enumerate(ranks):
            fig, (axes_row1, axes_row2) = \
            plt.subplots(2, len(error_types), figsize=(8, 5))

            axes_to_plot_layers = axes_row2
            axes_to_plot_baselines = axes_row1

            for error_type_i, error_type in enumerate(error_types):
                for output_layer in output_layers:
                    for metric in tracked_metrics:
                        if metric == 'ci':
                            ci_low = np.array(
                                results_collector[model_name][rank][error_type][output_layer]['ci'])[:, 0]
                            ci_high = np.array(
                                results_collector[model_name][rank][error_type][output_layer]['ci'])[:, 1]
                            axes_to_plot_layers[error_type_i].fill_between(
                                lesion_ratios,
                                ci_low,
                                ci_high,
                                alpha=0.3,
                                color='#DADADA',
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
                                    color = '#8B9FA5'
                                else: 
                                    color = '#9ABA79'
                                ax_to_plot = axes_to_plot_baselines[error_type_i]
                            else:
                                # for non-baseline layer performance,
                                # we label each layer and use layer-specific color.
                                label = output_layer
                                if "predictions" in label: label = "logits"
                                color = data.load_envs_dict(model_name, envs)[
                                    f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                                ax_to_plot = axes_to_plot_layers[error_type_i]
                            
                            # either baseline or non-baseline layer performance,
                            # we always plot them.
                            ax_to_plot.plot(
                                lesion_ratios,
                                results_collector[model_name][rank][error_type][output_layer][metric],
                                label=label,
                                color=color,
                                marker='o',
                            )
                
                axes_to_plot_baselines[error_type_i].spines['bottom'].set_visible(False)
                axes_to_plot_baselines[error_type_i].spines['top'].set_visible(False)
                axes_to_plot_baselines[error_type_i].spines['right'].set_visible(False)
                axes_to_plot_layers[error_type_i].spines['right'].set_visible(False)
                axes_to_plot_layers[error_type_i].spines['top'].set_visible(False)
                axes_to_plot_baselines[error_type_i].set_xticks([])
                if 'loc' in error_type: ylim = [0, 8]
                elif 'rot' in error_type: ylim = [0, 20]
                elif 'dist' in error_type: ylim = [0, 4]
                axes_to_plot_layers[error_type_i].set_ylim(ylim)

                # Add diagonal lines to connect the subplots
                d = 0.015  # How big to make the diagonal lines in axes coordinates
                kwargs = dict(transform=axes_to_plot_baselines[error_type_i].transAxes, color='k', clip_on=False)
                axes_to_plot_baselines[error_type_i].plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal line
                kwargs.update(transform=axes_to_plot_layers[error_type_i].transAxes)  # switch to the bottom subplot
                axes_to_plot_layers[error_type_i].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal line

                axes_to_plot_layers[error_type_i].set_xlabel('Lesion ratio')
                axes_to_plot_layers[error_type_i].set_xticks(lesion_ratios)
                axes_to_plot_layers[error_type_i].set_xticklabels(lesion_ratios)
                if rank == 'top':
                    if error_type == 'loc':  title = 'Top Coef. Lesion\n(Location)'
                    elif error_type == 'rot': title = 'Top Coef. Lesion\n(Direction)'
                    elif error_type == 'dist': title = 'Top Coef. Lesion\n(Nearest Border)'
                elif rank == 'random':
                    if error_type == 'loc':  title = 'Random Coef. Lesion\n(Location)'
                    elif error_type == 'rot': title = 'Random Coef. Lesion\n(Direction)'
                    elif error_type == 'dist': title = 'Random Coef. Lesion\n(Nearest Border)'
                axes_to_plot_baselines[error_type_i].set_title(title)

            plt.tight_layout()
            plt.legend(loc='upper right')
            plt.savefig(f'figs/paper/lesion_by_coef_{model_name}_rank={rank}.png')
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
    lesion_ratios = [0, 0.1, 0.3, 0.5, 0.7]

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
        
        fig, axes = plt.subplots(len(ranks), len(unit_chart_types), figsize=(20, 8))
        for rank_i, rank in enumerate(ranks):
            for unit_chart_type_i, unit_chart_type in enumerate(unit_chart_types):

                if unit_chart_type in ['maxvalueinclusters', 'numclusters']:
                    error_type = 'loc'
                elif unit_chart_type == 'directioness':
                    error_type = 'rot'
                elif unit_chart_type == 'borderness':
                    error_type = 'dist'

                for output_layer in output_layers:
                    for metric in tracked_metrics:
                        if metric == 'ci':
                            ci_low = np.array(
                                results_collector[model_name][rank][unit_chart_type][output_layer]['ci'])[:, 0]
                            ci_high = np.array(
                                results_collector[model_name][rank][unit_chart_type][output_layer]['ci'])[:, 1]
                            
                            # TEMP
                            ci_low = _convert_mse_to_physical_unit(ci_low, error_type, normalized=normalize_error)
                            ci_high = _convert_mse_to_physical_unit(ci_high, error_type, normalized=normalize_error)
                            
                            axes[rank_i, unit_chart_type_i].fill_between(
                                lesion_ratios,
                                ci_low,
                                ci_high,
                                alpha=1,
                                color='#DADADA',
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
                                    color = '#8B9FA5'
                                else: 
                                    color = '#9ABA79'
                            else:
                                # for non-baseline layer performance,
                                # we label each layer and use layer-specific color.
                                label = output_layers_2_levels[model_name][output_layer]
                                if "predictions" in label: label = "logits"
                                color = data.load_envs_dict(model_name, envs)[
                                    f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                            
                            # either baseline or non-baseline layer performance,
                            # we always plot them.
                            axes[rank_i, unit_chart_type_i].plot(
                                lesion_ratios,
                                # results_collector[model_name][rank][unit_chart_type][output_layer][metric],
                                # TEMP
                                _convert_mse_to_physical_unit(
                                    np.array(
                                        results_collector[model_name][rank][unit_chart_type][output_layer][metric]
                                    ), 
                                    error_type,
                                    normalized=normalize_error,
                                ),
                                label=label,
                                color=color,
                                marker='o',
                            )
                axes[rank_i, unit_chart_type_i].set_xlabel('Exclusion ratio')
                if normalize_error is True:
                    ax_label = 'Normalized Error'
                elif normalize_error is False:
                    ax_label = 'Error (physical)'
                else:
                    raise ValueError(f"Unknown value for normalize_error: {normalize_error}")
                if error_type in ['loc', 'dist']:
                    axes[rank_i, unit_chart_type_i].set_ylabel(ax_label)
                elif error_type == 'rot':
                    axes[rank_i, unit_chart_type_i].set_ylabel(ax_label, labelpad=10)
                axes[rank_i, unit_chart_type_i].set_xticks(lesion_ratios)
                axes[rank_i, unit_chart_type_i].set_xticklabels(lesion_ratios)
                if rank == 'top':
                    if unit_chart_type == 'maxvalueinclusters':
                        title = 'Top Place Field Activity\n(Location)'
                    elif unit_chart_type == 'numclusters':
                        title = 'Top Place Field Quantity\n(Location)'
                    elif unit_chart_type == 'directioness':
                        title = 'Top Directional Tuning\n(Direction)'
                    elif unit_chart_type == 'borderness':
                        title = 'Top Border Tuning\n(Nearest Border)'
                elif rank == 'random':
                    if unit_chart_type == 'maxvalueinclusters':
                        title = 'Random Place Field Activity\n(Location)'
                    elif unit_chart_type == 'numclusters':
                        title = 'Random Place Field Quantity\n(Location)'
                    elif unit_chart_type == 'directioness':
                        title = 'Random Directional Tuning\n(Direction)'
                    elif unit_chart_type == 'borderness':
                        title = 'Random Border Tuning\n(Nearest Border)'
                
                axes[rank_i, unit_chart_type_i].set_title(title, fontsize=18)
                axes[rank_i, unit_chart_type_i].spines.right.set_visible(False)
                axes[rank_i, unit_chart_type_i].spines.top.set_visible(False)
        
        axes[0, -1].legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        if normalize_error is True:
            plt.savefig(f'figs/paper/lesion_by_unit_chart_{model_name}.png')
            plt.savefig(f'figs/paper/lesion_by_unit_chart_{model_name}.svg')
        elif normalize_error is False:
            plt.savefig(f'figs/paper/lesion_by_unit_chart_{model_name}_unnorm.png')
            plt.savefig(f'figs/paper/lesion_by_unit_chart_{model_name}_unnorm.svg')
        else:
            raise ValueError(f"Unknown value for normalize_error: {normalize_error}")
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
    for unit_type in [
        'place_cell',
        'directional_cell',
        # 'border_cell',
        # 'place+directional_cell', 
    ]:
        if unit_type == 'place_cell':
            # Less directional place fields
            selected_n_indices = [1280, 2672, 315, 758, 3429, 3530]
            # subplot_label = 'A'
            # selected_n_indices = [88198, 128970, 41198, 128053, 147654, 35295]
            # selected_n_indices = [637, 968, 1292, 121, 1029, 1853, 728]
        
        if unit_type == 'directional_cell':
            # Less place fields directional cells
            selected_n_indices = [245, 1303, 1081, 646, 4013, 1499]
            # subplot_label = 'B'
            # selected_n_indices = [87232, 86, 86880, 4720, 104896, 142643]
            # selected_n_indices = [1829, 2031, 1013, 1667, 1679, 1553]

        if unit_type == 'border_cell':
            # border cells
            selected_n_indices = [3866, 2404, 1476, 2433, 3846, 2949]
            # subplot_label = 'C'
            pass
        
        if unit_type == 'place+directional_cell':
            # single field place cell with directional tuning
            selected_n_indices = [2631, 8, 2803, 1654, 475, 4055]
            # subplot_label = 'D'
            pass


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
                    ax.set_title(f'Unit {unit_index}')
                    # if col_index == 0:
                    #     # add subplot labels
                    #     ax.text(-0.4, 1.1, fontsize=14, fontweight='bold',
                    #     transform=ax.transAxes, va='top', ha='left')

                    # --- subplot2: plot polar plot for the selected units ---
                    ax = fig.add_subplot(gs[1, col_index], projection='polar')
                    # we want to have a closed loop
                    theta = np.linspace(0, 2*np.pi, model_reps.shape[1], endpoint=False)
                    x = theta.tolist() + [theta[0]]
                    y = per_rotation_vector_length[unit_index] + [per_rotation_vector_length[unit_index][0]]
                    ax.plot(x, y)
                    ax.set_theta_zero_location("N")
                    ax.set_theta_direction(-1)
                    # ax.set_rticks([])
                    if col_index == 0:
                        ax.set_thetagrids([0, 90, 180, 270], labels=['0', '90', '180', '270'])
                    else:
                        ax.set_thetagrids([0, 90, 180, 270], labels=['', '', '', ''])
        
        # add text on bottom middle overall as title
        # if unit_type == 'place_cell':
        #     suptitle = 'Place Cells'
        # elif unit_type == 'directional_cell':
        #     suptitle = 'Directional Cells'
        # elif unit_type == 'border_cell':
        #     suptitle = 'Border Cells'
        # elif unit_type == 'place+directional_cell':
        #     suptitle = 'Place + Directional Cells'
        # plt.text(0.5, 1.25, suptitle=suptitle, ha='center', va='center', fontsize=16, transform=fig.transFigure)
        # plt.savefig(f'figs/paper/unit_visualization_by_type_{unit_type}.png')
        plt.savefig(f'figs/paper/unit_visualization_by_type_{unit_type}_{config_version}.png')
        plt.close()
      

def unit_chart_visualization_piechart():
    experiment = 'unit_chart'
    moving_trajectory = 'uniform'
    movement_mode = '2d'
    envs = ['env28_r24']
    model_names = [
        'vgg16', 
        'resnet50', 'vit_b16', 'vgg16_untrained', 'resnet50_untrained', 'vit_b16_untrained'
    ]

    for env in envs:
        for model_name in model_names:
            output_layers = data.load_model_layers(model_name)

            # each column is a layer
            # row1 is all types piechart
            fig = plt.figure(figsize=(24, 5))
            gs = fig.add_gridspec(
                nrows=1, ncols=len(output_layers)
            )

            for col_index, output_layer in enumerate(output_layers):
                config_version = f'{env}_{movement_mode}_{model_name}_{output_layer}'
                print(f'Plotting {config_version}')
                config = utils.load_config(config_version)
                
                # load unit chart info
                results_path = utils.load_results_path(
                    config=config,
                    experiment=experiment,
                    moving_trajectory=moving_trajectory,
                )
                unit_chart_info = np.load(
                    f'{results_path}/unit_chart.npy', allow_pickle=True)

                # dict by type
                unit_indices_by_types = umc._unit_chart_type_classification(unit_chart_info)

                # first subplot: piechart of different cell proportions
                n_exc_place_cells = len(unit_indices_by_types["exclusive_place_cells_indices"])
                n_exc_border_cells = len(unit_indices_by_types["exclusive_border_cells_indices"])
                n_exc_direction_cells = len(unit_indices_by_types["exclusive_direction_cells_indices"])
                n_place_and_border_not_direction_cells = len(unit_indices_by_types["place_and_border_not_direction_cells_indices"])
                n_place_and_direction_not_border_cells = len(unit_indices_by_types["place_and_direction_not_border_cells_indices"])
                n_border_and_direction_not_place_cells = len(unit_indices_by_types["border_and_direction_not_place_cells_indices"])
                n_place_border_direction_cells = len(unit_indices_by_types["place_border_direction_cells_indices"])
                n_active_no_type_cells = len(unit_indices_by_types["active_no_type_indices"])

                sum_n_cells = \
                    n_exc_place_cells + \
                    n_exc_border_cells + \
                    n_exc_direction_cells + \
                    n_place_and_border_not_direction_cells + \
                    n_place_and_direction_not_border_cells + \
                    n_border_and_direction_not_place_cells + \
                    n_place_border_direction_cells + \
                    n_active_no_type_cells

                labels = []
                n_cells = []

                # collect the number of cells for each type
                # only if n of cells > 0 and collect the labels
                if n_exc_place_cells/sum_n_cells >= 0.01:
                    n_cells.append(n_exc_place_cells)
                    labels.append('P')

                if n_exc_border_cells/sum_n_cells >= 0.01:
                    n_cells.append(n_exc_border_cells)
                    labels.append('B')

                if n_exc_direction_cells/sum_n_cells >= 0.01:
                    n_cells.append(n_exc_direction_cells)
                    labels.append('D')

                if n_place_and_border_not_direction_cells/sum_n_cells >= 0.01:
                    n_cells.append(n_place_and_border_not_direction_cells)
                    labels.append('P+B')

                if n_place_and_direction_not_border_cells/sum_n_cells >= 0.01:
                    n_cells.append(n_place_and_direction_not_border_cells)
                    labels.append('P+D')

                if n_border_and_direction_not_place_cells/sum_n_cells >= 0.01:
                    n_cells.append(n_border_and_direction_not_place_cells)
                    labels.append('B+D')

                if n_place_border_direction_cells/sum_n_cells >= 0.01:
                    n_cells.append(n_place_border_direction_cells)
                    labels.append('P+B+D')
                
                if n_active_no_type_cells/sum_n_cells >= 0.01:
                    n_cells.append(n_active_no_type_cells)
                    labels.append('Active (no type)')
                
                # And lastly, the dead units
                n_cells.append(len(unit_indices_by_types["dead_units_indices"]))
                labels.append('Inactive')
                
                # make sure plt.cm.Pastel1.colors are consistent across layers
                # for each type of cells.
                colors = []
                for label in labels:
                    if label == 'P':
                        colors.append(plt.cm.Pastel1.colors[1])
                    elif label == 'B':
                        colors.append(plt.cm.Pastel1.colors[0])
                    elif label == 'D':
                        colors.append(plt.cm.Pastel1.colors[2])
                    elif label == 'P+B':
                        colors.append(plt.cm.Pastel1.colors[3])
                    elif label == 'P+D':
                        colors.append(plt.cm.Pastel1.colors[4])
                    elif label == 'B+D':
                        colors.append(plt.cm.Pastel1.colors[5])
                    elif label == 'P+B+D':
                        colors.append(plt.cm.Pastel1.colors[6])
                    elif label == 'Inactive':
                        colors.append("grey")
                    elif label == 'Active (no type)':
                        colors.append(plt.cm.Pastel1.colors[7])

                ax = fig.add_subplot(gs[col_index])

                # Calculate percentages and exclude labels with 0 percentage
                total_cells = sum(n_cells)
                percentages = [round((cell / total_cells) * 100) for cell in n_cells]
                filtered_labels = [
                    label if percentage > 0 else '' \
                        for label, percentage in zip(labels, percentages)
                ]

                ax.pie(
                    n_cells,
                    autopct=lambda p: '{:.0f}'.format(round(p)) if p >= 1 else '',
                    labels=filtered_labels,
                    colors=colors,
                    explode=[0.1]*len(labels),
                    textprops={'fontsize': 14},
                )

                if 'vgg16' in model_name:
                    if 'untrained' in model_name:
                        model_name_plot = 'VGG-16 (untrained)'
                    else:
                        model_name_plot = 'VGG-16'
                    if output_layer == 'block2_pool':
                        output_layer_plot = 'Early (block2_pool)'
                    elif output_layer == 'block4_pool':
                        output_layer_plot = 'Mid (block4_pool)'
                    elif output_layer == 'block5_pool':
                        output_layer_plot = 'Late (block5_pool)'
                    elif output_layer == 'fc2':
                        output_layer_plot = 'Penultimate (fc2)'

                elif 'resnet50' in model_name:
                    if 'untrained' in model_name:
                        model_name_plot = 'ResNet-50 (untrained)'
                    else:
                        model_name_plot = 'ResNet-50'
                    if output_layer == 'conv2_block3_out':
                        output_layer_plot = 'Early (conv2_block3_out)'
                    elif output_layer == 'conv4_block6_out':
                        output_layer_plot = 'Mid (conv4_block6_out)'
                    elif output_layer == 'conv5_block2_out':
                        output_layer_plot = 'Late (conv5_block2_out)'
                    elif output_layer == 'avg_pool':
                        output_layer_plot = 'Penultimate (avg_pool)'

                elif 'vit_b16' in model_name:
                    if 'untrained' in model_name:
                        model_name_plot = 'ViT-B/16 (untrained)'
                    else:
                        model_name_plot = 'ViT-B/16'
                    if output_layer == 'layer_3':
                        output_layer_plot = 'Early (layer_3)'
                    elif output_layer == 'layer_6':
                        output_layer_plot = 'Mid (layer_6)'
                    elif output_layer == 'layer_9':
                        output_layer_plot = 'Late (layer_9)'
                    elif output_layer == 'layer_12':
                        output_layer_plot = 'Penultimate (layer_12)'

                ax.set_title(f'{output_layer_plot}', fontweight='bold', fontsize=22)

                # remove left, top and right
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # make subplot space wider
                plt.subplots_adjust(wspace=0.8)

                plt.suptitle(f'{model_name_plot}', fontweight='bold', fontsize=22)
                plt.savefig(f'figs/paper/unit_chart_overlaps_{model_name}_{env}.png')
                plt.savefig(f'figs/paper/unit_chart_overlaps_{model_name}_{env}.pdf')


def unit_chart_against_coef_for_all_units():
    """
    Show relationship of units between their unit chart info and decoder coef values.
    This complements the lesion analysis by more directly and comprehensively showing
    how they may differ (we know they should given lesion results).

    Implementation:
        For each model's each layer, load the corresponding unit chart info
        (from `unit_chart.npy`) and coef info (from `res.npy`).

        Then, scatter plot the unit chart info against coef info for all units. 
        - Unit chart info for each subplot should include each unit's 
            `maxvalueinclusters`, `numclusters`, `directioness`, `borderness`.
        - coef info should include each unit's coef value.

        Specifically,
        - `unit_chart_info` \in (n_units, unit_chart_dims)
            where `maxvalueinclusters` is column 3, `numclusters` is column 1,
            `directioness` is column 10, `borderness` is column 9.
        - `coef` \in (3, n_units) if `experiment=loc_n_rot` 
            or \in (1, n_units) if `experiment=border_dist`

            if `experiment=loc_n_rot`, coef is [coef_x, coef_y, coef_rot],
            when plotting, we do coef = np.abs(coef) and coef_loc = np.mean(coef[:2, :], axis=0)
            to be consistent with `lesion.py`

            if `experiment=border_dist`, coef is [coef_dist]

    Figure layout:
        - Each figure is a model's layer at a sampling rate and random seed.
        - Each row is a unit_chart_dim, each column is this unit chart type vs. coef * 3 (loc, rot, dist).
    """
    env = "env28_r24"
    movement_mode = '2d'
    sampling_rates = [0.3]
    random_seeds = [42]
    model_names = [
        'vgg16', 
        'resnet50',
        'vit_b16',
        'vgg16_untrained',
        'resnet50_untrained',
        'vit_b16_untrained',
    ]
    moving_trajectory = 'uniform'
    decoding_model_choice = {'name': 'ridge_regression', 'hparams': 1.0}
    feature_selection = 'l2'
    unit_chart_dims = {
        'maxvalueinclusters': 3,
        'numclusters': 1,
        'directioness': 10,
        'borderness': 9,
    }

    for sampling_rate in sampling_rates:
        for random_seed in random_seeds:
            for model_name in model_names:
                output_layers = data.load_model_layers(model_name)
                                
                for output_layer in output_layers:
                    config_version = f'{env}_{movement_mode}_{model_name}_{output_layer}'
                    config = utils.load_config(config_version)

                    # load coef for `loc_n_rot`
                    results_path = utils.load_results_path(
                        config=config,
                        experiment="loc_n_rot",
                        feature_selection=feature_selection,
                        decoding_model_choice=decoding_model_choice,
                        sampling_rate=sampling_rate,
                        moving_trajectory=moving_trajectory,
                        random_seed=random_seed,
                    )
                    # \in (3, n_units)
                    coef_loc_n_rot = np.load(f'{results_path}/res.npy', allow_pickle=True).item()['coef']
                    print(f"coef_loc_n_rot.shape: {coef_loc_n_rot.shape}")
                    # average over x and y and separate coef_loc and coef_rot
                    coef_loc_n_rot = np.abs(coef_loc_n_rot)
                    coef_loc = np.mean(coef_loc_n_rot[:2, :], axis=0)  # (n_units,)
                    coef_rot = coef_loc_n_rot[2, :]    # (n_units,)

                    # load coef for `border_dist`
                    results_path = utils.load_results_path(
                        config=config,
                        experiment="border_dist",
                        feature_selection=feature_selection,
                        decoding_model_choice=decoding_model_choice,
                        sampling_rate=sampling_rate,
                        moving_trajectory=moving_trajectory,
                        random_seed=random_seed,
                    )
                    # \in (1, n_units)
                    coef_dist = np.load(f'{results_path}/res.npy', allow_pickle=True).item()['coef']
                    coef_dist = np.abs(coef_dist)
                    print(f"coef_dist.shape: {coef_dist.shape}")

                    # load unit chart info
                    results_path = utils.load_results_path(
                        config=config,
                        experiment="unit_chart",
                        moving_trajectory=moving_trajectory,
                    )
                    # \in (n_units, unit_chart_dims)
                    unit_chart_info = np.load(
                        f'{results_path}/unit_chart.npy', allow_pickle=True)
                    print(f"unit_chart_info.shape: {unit_chart_info.shape}")

                    fig, axes = plt.subplots(len(unit_chart_dims), 3, figsize=(15, 5 * len(unit_chart_dims)))
                    for row_i, (unit_chart_dim, unit_chart_dim_index) in enumerate(unit_chart_dims.items()):
                        unit_chart_dim_info = unit_chart_info[:, unit_chart_dim_index]

                        print(f"unit_chart_dim: {unit_chart_dim}")
                        print(unit_chart_dim_info)

                        if unit_chart_dim == 'maxvalueinclusters':
                            # due to multiple fields, need global max per unit
                            unit_chart_dim_info = [np.max(per_unit_fields) for per_unit_fields in unit_chart_dim_info]
                            xlabel = "Place Field Activity"
                        elif unit_chart_dim == 'numclusters':
                            unit_chart_dim_info = [np.max(per_unit_fields) for per_unit_fields in unit_chart_dim_info]
                            xlabel = "Place Field Quantity"
                        elif unit_chart_dim == 'directioness':
                            xlabel = "Directional Tuning"
                        elif unit_chart_dim == 'borderness':
                            xlabel = "Border Tuning"
                        else:
                            raise ValueError(f"Unknown unit_chart_dim: {unit_chart_dim}")
                        
                        for col_i, ax in enumerate(axes[row_i]):
                            print(f"row_i: {row_i}, col_i: {col_i}")
                            if col_i == 0:
                                coef = coef_loc
                                ylabel = 'coef. (location)'
                            elif col_i == 1:
                                coef = coef_rot
                                ylabel = 'coef. (rotation)'
                            else:
                                coef = coef_dist[0, :]
                                ylabel = 'coef. (border dist.)'

                            ax.scatter(
                                unit_chart_dim_info,
                                coef,
                                alpha=0.2,
                                edgecolors='b',
                                color='blue',
                            )
                            ax.set_xlabel(xlabel)
                            ax.set_ylabel(ylabel)
                            ax.spines['right'].set_visible(False)
                            ax.spines['top'].set_visible(False)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.suptitle(
                        f"{model_names_2_pretty_names[model_name]} - "
                        f"{output_layers_2_levels[model_name][output_layer]}", 
                        fontsize=18
                    )
                    plt.savefig(
                        f'figs/paper/unit_chart_against_coef_{model_name}_{output_layer}'
                        f'_sr{sampling_rate}_seed{random_seed}.png'
                    )
                    plt.close()
                        

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    TF_NUM_INTRAOP_THREADS = 10
    normalize_error = False
    # decoding_each_model_across_layers_and_sr()
    # decoding_all_models_one_layer_one_sr()
    # lesion_by_coef_each_model_across_layers_and_lr()
    # lesion_by_unit_chart_each_model_across_layers_and_lr()
    # unit_visualization_by_type()
    # unit_chart_visualization_piechart()

    unit_chart_against_coef_for_all_units()
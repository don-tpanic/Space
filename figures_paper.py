from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import data


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


def decoding_all_models_one_layer_one_sr():
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


def lesion_by_coef_each_model_across_layers_and_lr():
    envs = ['env28_r24']
    env = envs[0]
    movement_mode = '2d'
    sampling_rate = 0.3
    random_seeds = [42]
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
        plt.savefig(f'figs/paper/lesion_{model_name}.png')
        plt.close()


if __name__ == '__main__':
    # decoding_each_model_across_layers_and_sr()
    # decoding_all_models_one_layer_one_sr()
    lesion_by_coef_each_model_across_layers_and_lr()
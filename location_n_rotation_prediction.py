import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import time
import logging
import itertools
import multiprocessing
from collections import defaultdict

import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import utils
import data
import models
import lesion

"""
Experiment script
    Location and rotation prediction.
    DNN model representations are fit on a linear regression model
    to predict agent's locations and rotations.
"""

def _determine_moving_trajectory(
        moving_trajectory,
        results_path,
        n_rotations,
        sampling_rate, 
        model_reps, 
        targets_true,
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        random_seed,
    ):
    """
    While the real data is captured by the agent moving uniformly
    on a grid in Unity, we could manipulate the split of train/test
    to imitate different moving trajectories. This could be used to 
    investigate how well the model can generalise to unseen data (
    i.e. interpolation vs extrapolation).
    """
    if moving_trajectory == 'uniform':
        # make sure a sampled loc's all rotates are in train
        # so the sampling indices need to be locs of all views
        np.random.seed(random_seed)
        train_sample_loc_indices = np.random.choice(
            model_reps.shape[0] // n_rotations,
            size=int(sampling_rate * model_reps.shape[0] // n_rotations),
            replace=False,
        )

        # the actual sampling indices need to be adjusted
        # to include all rotations of the sampled locs
        # in other words, each index in sampled_loc_indices
        # needs to be incremented n_rotations times
        # e.g. if sampled_loc_indices = [0, 1, 2] and 
        # n_rotations = 2, then sampled_indices = [0,1, 2,3, 4,5]
        train_sample_indices = []
        for i in train_sample_loc_indices:
            train_sample_indices.extend(
                [i*n_rotations + j for j in range(n_rotations)]
            )

    elif moving_trajectory == 'uniform_loc_random_rot':
        # sampled locations' rotations are randomly sampled
        # into training.
        np.random.seed(random_seed)
        train_sample_indices = np.random.choice(
            model_reps.shape[0],
            size=int(sampling_rate * model_reps.shape[0]),
            replace=False,
        )
        
    # now we can use the sampled indices to get the train/test data
    X_train = model_reps[train_sample_indices, :]
    y_train = targets_true[train_sample_indices, :]
    X_test = np.delete(model_reps, train_sample_indices, axis=0)
    y_test = np.delete(targets_true, train_sample_indices, axis=0)

    if logging_level == 'debug':
        # DEBUG: exploding coef at higher sampling rate
        # save X_train to disk to `results_path` and named
        # based on current sampling_rate
        np.save(f'{results_path}/X_train_{sampling_rate}.npy', X_train)

    del model_reps
    return X_train, X_test, y_train, y_test


def _load_train_test_data(
        model,
        config,
        preprocessed_data,
        targets_true,
        moving_trajectory,
        sampling_rate,
        feature_selection,
        decoding_model_choice,
        results_path,
        random_seed,
    ):
    """
    Given the entire dataset produced by a given model+layer,
    produce the train/test data based on the moving trajectory,
    sampling rate, and random seed.

    The splitted data are then used by `fit_decoding_model`.

    return:
        X_train, X_test, y_train, y_test
    """
    model_reps = data.load_full_dataset_model_reps(
        config=config, model=model, 
        preprocessed_data=preprocessed_data,
    )

    # TODO: remember after lesion, the meaning of the columns change; further analysis 
    # of coef needs to be careful.
    if 'lesion' in feature_selection:
        model_reps = lesion.lesion(
            config=config,
            moving_trajectory=moving_trajectory,
            feature_selection=feature_selection,
            model_reps=model_reps,
            reference_experiment=reference_experiment,
            decoding_model_choice=decoding_model_choice,
            sampling_rate=sampling_rate,
            random_seed=random_seed,
        )

    X_train, X_test, y_train, y_test = \
        _determine_moving_trajectory(
            model_reps=model_reps,
            targets_true=targets_true,
            moving_trajectory=moving_trajectory,
            results_path=results_path,
            n_rotations=config['n_rotations'],
            sampling_rate=sampling_rate,
            random_seed=random_seed,
            env_x_min=config['env_x_min'],
            env_x_max=config['env_x_max'],
            env_y_min=config['env_y_min'],
            env_y_max=config['env_y_max'],
        )
    logging.info(f'X_train.shape: {X_train.shape}')
    logging.info(f'X_test.shape: {X_test.shape}')
    return X_train, X_test, y_train, y_test


def _compute_per_loc_mse_rot_samples(
        y_test,
        y_pred,
        n_rotations,
    ):
    """
    Compute rotation error with MSE for each data-point.
    The thing here is to take into account that 
    predicting 15 degrees and 345 degrees while ground truth
    is 0 degrees should have the same error.
    Notice the y_test here are integers from 0 to n_rotations-1.
    So when computing MSE for each pair of predict and true, we 
    must consider if their difference is more than half of n_rotations
    (i.e. 180 degrees). If so, we need to convert the difference
    to the other side of the circle (i.e. 360 - diff).
    """
    # check if y_test and y_pred same length
    # if not meaning we are in baseline mode, 
    # where the y_pred should be repeated 
    # to be the same length as y_test
    if len(y_test) != len(y_pred):
        assert len(y_pred) == 1
        y_pred = np.repeat(y_pred, len(y_test))

    rotation_error = np.empty(len(y_test))
    for i in range(len(y_test)):
        y_test_i = y_test[i]
        y_pred_i = y_pred[i]
        diff = abs(y_test_i - y_pred_i)
        if diff > n_rotations / 2:
            diff = n_rotations - diff
        rotation_error[i] = diff**2
    return rotation_error


def _fit_decoding_model(
        X_train,
        X_test,
        y_train,
        y_test,
        config,
        decoding_model_choice,
        random_seed
    ):
    """
    The base case fitting decoding model to data and 
    produces a set of base-case result for saving.

    return:
        mse_loc, mse_rot, ci_loc, ci_rot, \
            decoding_model.coef_, decoding_model.intercept_, \
                baseline_predict_mid_mse_loc, baseline_predict_mid_mse_rot, \
                    baseline_predict_random_mse_loc, baseline_predict_random_mse_rot

    """
    logging.info(f'[Check] Fitting regression model..')

    if decoding_model_choice['name'] == 'linear_regression':
        decoding_model = linear_model.LinearRegression()
    elif decoding_model_choice['name'] == 'ridge_regression':
        decoding_model = linear_model.Ridge(
            alpha=decoding_model_choice['hparams'])
    elif decoding_model_choice['name'] == 'lasso_regression':
        decoding_model = linear_model.Lasso(
            alpha=decoding_model_choice['hparams'])
    decoding_model.fit(X_train, y_train)
    y_pred = decoding_model.predict(X_test)

    # compute errors ------------------------------------
    per_loc_mse_loc_samples = np.mean(
        np.square(y_test[:, :2] - y_pred[:, :2]), axis=1
    )
    per_loc_mse_rot_samples = _compute_per_loc_mse_rot_samples(
        y_test[:, 2:], y_pred[:, 2:], config['n_rotations'])

    logging.info(
        f'[Check] per_loc_mse_loc_samples.shape='\
        f'{per_loc_mse_loc_samples.shape}'
    )

    logging.info(
        f'[Check] per_loc_mse_rot_samples.shape='\
        f'{per_loc_mse_rot_samples.shape}'
    )

    mse_loc = np.mean(per_loc_mse_loc_samples)
    mse_rot = np.mean(per_loc_mse_rot_samples)
    ci_loc = stats.bootstrap((per_loc_mse_loc_samples,), np.mean).confidence_interval
    ci_rot = stats.bootstrap((per_loc_mse_rot_samples,), np.mean).confidence_interval
    logging.info('[Check] model results done')

    # baseline 1, predict mid ------------------------------------
    mid_loc = np.array(
        [(config['env_x_max']+config['env_x_min']), 
        (config['env_y_max']+config['env_y_min'])]
    )
    mid_rot = np.array([config['n_rotations']//4])

    baseline_predict_mid_mse_loc = \
        np.mean(
            np.square(y_test[:, :2] - mid_loc)
        )

    baseline_predict_mid_mse_rot = \
        np.mean(
            _compute_per_loc_mse_rot_samples(
                y_test[:, 2:], mid_rot, config['n_rotations']
            )
        )
    
    logging.info('[Check] baseline 1 done')

    # baseline error 2, predict random ------------------------------------
    # first, we sample random locations based on bounds of the env
    np.random.seed(random_seed)
    random_loc = np.random.uniform(
        low=np.array([config['env_x_min'], config['env_y_min']]),
        high=np.array([config['env_x_max'], config['env_y_max']]),
        size=(y_test.shape[0], 2)
    )
    # second, we sample random rotations
    random_rot = np.random.randint(
        low=0, high=config['n_rotations'], 
        size=(y_test.shape[0], 1)
    )

    baseline_predict_random_mse_loc = \
        np.mean(
            np.square(y_test[:, :2] - random_loc)
        )
    
    baseline_predict_random_mse_rot = \
        np.mean(
            _compute_per_loc_mse_rot_samples(
                y_test[:, 2:], random_rot, config['n_rotations']
            )
        )

    logging.info('[Check] baseline 2 done')
    return mse_loc, mse_rot, ci_loc, ci_rot, \
        decoding_model.coef_, decoding_model.intercept_, \
                baseline_predict_mid_mse_loc, baseline_predict_mid_mse_rot, \
                    baseline_predict_random_mse_loc, baseline_predict_random_mse_rot


def _single_env_decoding_error(
        config_version, 
        moving_trajectory,
        sampling_rate,
        experiment,
        feature_selection,
        decoding_model_choice,
        random_seed,
    ):
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{TF_NUM_INTRAOP_THREADS}"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    # logging.info(f'[Check] config_version: {config_version}')

    # check if feature_selection and decoding_model_choice match
    # specifically, we make sure if l1 is included in feature_selection,
    # decoding model must be lasso;
    # while this check seems strange but this is for future compatibility
    # where we have a metric-based feature selection while using lasso 
    # at the same time.
    if \
        (
            'l1' in feature_selection and \
            decoding_model_choice['name'] != 'lasso_regression'
        ) \
        or \
        (
            'l2' in feature_selection and \
            decoding_model_choice['name'] != 'ridge_regression'
        ):
        # logging.info(
        #     '[Skip] feature_selection and decoding_model_choice mismatch'
        # )
        return

    config = utils.load_config(config_version)
    results_path = utils.load_results_path(
        config=config,
        experiment=experiment,
        feature_selection=feature_selection,
        decoding_model_choice=decoding_model_choice,
        sampling_rate=sampling_rate,
        moving_trajectory=moving_trajectory,
        random_seed=random_seed
    )
    
    # check if this base-case result exists, 
    # if so skip
    if os.path.exists(f'{results_path}/res.npy'):
        # logging.info('[Check] base-case exists, skipping')
        return
    else:
        logging.info(
            f'[Begin job] {config_version}, {sampling_rate},'\
            f'{feature_selection}, {decoding_model_choice}, {random_seed}'
        )
        if config['model_name'] == 'none':
            model = None
            preprocess_func = None
        else:
            model, preprocess_func = models.load_model(
                config['model_name'], config['output_layer'])

        preprocessed_data = data.load_preprocessed_data(
            config=config,
            data_path=\
                f"data/unity/{config['unity_env']}/"\
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

        targets_true = data.load_decoding_targets(
            movement_mode=config['movement_mode'],
            env_x_min=config['env_x_min'],
            env_x_max=config['env_x_max'],
            env_y_min=config['env_y_min'],
            env_y_max=config['env_y_max'],
            multiplier=config['multiplier'],
            n_rotations=config['n_rotations'],
        )

        # for sampling_rate in sampling_rates:
        X_train, X_test, y_train, y_test = \
            _load_train_test_data(
                model=model,
                config=config,
                preprocessed_data=preprocessed_data,
                targets_true=targets_true,
                moving_trajectory=moving_trajectory,
                sampling_rate=sampling_rate,
                feature_selection=feature_selection,
                decoding_model_choice=decoding_model_choice,
                results_path=results_path,
                random_seed=random_seed,
            )
        
        mse_loc, mse_rot, ci_loc, ci_rot, coef, intercept, \
            baseline_predict_mid_mse_loc, baseline_predict_mid_mse_rot, \
                baseline_predict_random_mse_loc, baseline_predict_random_mse_rot = \
                    _fit_decoding_model(
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        config=config,
                        decoding_model_choice=decoding_model_choice,
                        random_seed=random_seed,
                    )

        res = defaultdict(dict)
        res['loc']['mse'] = mse_loc
        res['rot']['mse'] = mse_rot
        res['loc']['ci'] = ci_loc
        res['rot']['ci'] = ci_rot
        res['loc']['baseline_predict_mid_mse'] = baseline_predict_mid_mse_loc
        res['loc']['baseline_predict_random_mse'] = baseline_predict_random_mse_loc
        res['rot']['baseline_predict_mid_mse'] = baseline_predict_mid_mse_rot
        res['rot']['baseline_predict_random_mse'] = baseline_predict_random_mse_rot
        res['coef'] = coef
        res['intercept'] = intercept
        np.save(f'{results_path}/res.npy', res)


def multi_envs_across_dimensions_CPU(
        target_func,
        envs,
        model_names,
        experiment,
        moving_trajectories,
        sampling_rates,
        feature_selections,
        decoding_model_choices,
        random_seeds,
    ):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with multiprocessing.Pool(processes=CPU_NUM_PROCESSES) as pool:
        for model_name in model_names:
            envs_dict = data.load_envs_dict(model_name, envs)
            config_versions=list(envs_dict.keys())
            for config_version in config_versions:
                for moving_trajectory in moving_trajectories:
                    for sampling_rate in sampling_rates:
                        for feature_selection in feature_selections:
                            for decoding_model_choice in decoding_model_choices:
                                for random_seed in random_seeds:
                                    res = pool.apply_async(
                                        target_func,
                                        args=(
                                            config_version, 
                                            moving_trajectory,
                                            sampling_rate,
                                            experiment,
                                            feature_selection,
                                            decoding_model_choice,
                                            random_seed,
                                        )
                                    )
        logging.info(res.get())
        pool.close()
        pool.join()


def multi_envs_across_dimensions_GPU(
        target_func,
        envs,
        experiment,
        sampling_rates,
        model_names,
        moving_trajectories,
        decoding_model_choices,
        feature_selections,
        random_seeds,
        cuda_id_list=[0, 1, 2, 3, 4, 5, 6, 7],
    ):
    for model_name in model_names:
        envs_dict = data.load_envs_dict(model_name, envs)
        config_versions=list(envs_dict.keys())
        args_list = []
        for config_version in config_versions:
            for moving_trajectory in moving_trajectories:
                for sampling_rate in sampling_rates:
                    for feature_selection in feature_selections:
                        for decoding_model_choice in decoding_model_choices:
                            for random_seed in random_seeds:
                                single_entry = {}
                                single_entry['config_version'] = config_version
                                single_entry['moving_trajectory'] = moving_trajectory
                                single_entry['sampling_rate'] = sampling_rate
                                single_entry['experiment'] = experiment
                                single_entry['feature_selection'] = feature_selection
                                single_entry['decoding_model_choice'] = decoding_model_choice
                                single_entry['random_seed'] = random_seed
                                args_list.append(single_entry)

    logging.info(f'args_list = {args_list}')
    logging.info(f'args_list len = {len(args_list)}')
    utils.cuda_manager(
        target_func, args_list, cuda_id_list
    )


def cross_dimension_analysis(
        analysis='across_sampling_rates_n_layers',
        envs=['env28_r24'],
        movement_modes=['2d'],
        model_names=['resnet50'],
        moving_trajectories=['uniform'],
        feature_selections=['l2'],
        sampling_rates=[0.01, 0.5],
        decoding_model_choices=[{'name': 'ridge_regression', 'hparams': 1.0}],
        random_seeds=[42, 1234],
        experiment='loc_n_rot'
    ):
    """
    Dimension here refers to sampling rate, layers, etc.
    So cross-dimension analysis could be that we are interested in
    for the same [model, feature selection, decoding model, etc], how
    does decoding performance vary across sampling rates and layers.

    There are other combination of dimensions one can analyse such as 
    for the same [sampling rate, layer, etc] but analyse different
    decoding models.

    This function is meant to be general enough to handle possible 
    cross-dimension analysis without having to sweat on extracting the 
    right dimensions with a lot of extra code.

    Under the hood, this function aggregates the base-case results 
    produced by `single_env_decoding_error` based on the dimension 
    combinations of interest.
    """
    error_types = ['loc', 'rot']
    tracked_metrics = ['mse', 'ci', 
        'baseline_predict_mid_mse', 'baseline_predict_random_mse']
    tracked_regression_weights = ['coef', 'intercept']

    if analysis == 'decoding_across_sampling_rates_n_layers':

        env = envs[0]
        moving_trajectory = moving_trajectories[0]
        movement_mode = movement_modes[0]

        for model_name in model_names:
            output_layers = data.load_model_layers(model_name)
            for feature_selection in feature_selections:
                for decoding_model_choice in decoding_model_choices:
                    decoding_model_name = decoding_model_choice['name']
                    decoding_model_hparams = decoding_model_choice['hparams']

                    if \
                        (
                            'l1' in feature_selection and \
                            decoding_model_choice['name'] != 'lasso_regression'
                        ) \
                        or \
                        (
                            'l2' in feature_selection and \
                            decoding_model_choice['name'] != 'ridge_regression'
                        ):
                        continue

                    # collect results across dimensions
                    # from base-case results.
                    results_collector = \
                        defaultdict(                            # key - error_type
                            lambda: defaultdict(                # key - output_layer
                                lambda: defaultdict(list)       # key - metric
                            )
                        )
                    
                    for error_type in error_types:
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
                    # left subplot for loc error, right subplot for rot error.
                    # x-axis is sampling rate, y-axis is decoding error.
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
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
                                        alpha=0.2,
                                        color='grey',
                                    )
                                else:
                                    if 'baseline' in metric:
                                        # no need to label baseline for each layer
                                        # we only going to label baseline when we plot
                                        # the last layer.
                                        if output_layer == output_layers[-1]:
                                            label = metric
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
                                    )
                        axes[i].set_xlabel('sampling rates')
                        axes[i].set_title(error_type)
                        axes[i].grid()
                    sup_title = f'{envs[0]},{movement_mode},'\
                                f'{model_name},{feature_selection},'\
                                f'{decoding_model_name}({decoding_model_hparams})'
                    # for across layers and sampling rates, 
                    # we save the plot at the same level as layers.
                    figs_path = f'figs/{env}/{movement_mode}/{moving_trajectory}/'\
                                    f'{model_name}/{experiment}/{feature_selection}/'\
                                    f'{decoding_model_name}_{decoding_model_hparams}'
                    if not os.path.exists(figs_path):
                        os.makedirs(figs_path)
                    plt.legend()
                    plt.suptitle(sup_title)
                    plt.savefig(f'{figs_path}/decoding_across_sampling_rates_n_layers.png')
                    plt.close()
                    logging.info(f'[Saved] {figs_path}/decoding_across_sampling_rates_n_layers.png')

    elif analysis == 'decoding_across_sampling_rates_n_layers_per_seed':

        env = envs[0]
        moving_trajectory = moving_trajectories[0]
        movement_mode = movement_modes[0]

        for random_seed in random_seeds:
            for model_name in model_names:
                output_layers = data.load_model_layers(model_name)
                for feature_selection in feature_selections:
                    for decoding_model_choice in decoding_model_choices:
                        decoding_model_name = decoding_model_choice['name']
                        decoding_model_hparams = decoding_model_choice['hparams']

                        if \
                            (
                                'l1' in feature_selection and \
                                decoding_model_choice['name'] != 'lasso_regression'
                            ) \
                            or \
                            (
                                'l2' in feature_selection and \
                                decoding_model_choice['name'] != 'ridge_regression'
                            ):
                            continue

                        # collect results across dimensions
                        # from base-case results.
                        results_collector = \
                            defaultdict(                            # key - error_type
                                lambda: defaultdict(                # key - output_layer
                                    lambda: defaultdict(list)       # key - metric
                                )
                            )
                        
                        for error_type in error_types:
                            for output_layer in output_layers:
                                # we accumulate results across sampling rates in the base-list
                                # of the nested defaultdict.
                                for sampling_rate in sampling_rates:
                                    results_path = \
                                        f'results/{env}/{movement_mode}/{moving_trajectory}/'\
                                        f'{model_name}/{experiment}/{feature_selection}/'\
                                        f'{decoding_model_name}_{decoding_model_hparams}/'\
                                        f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
                                    results = np.load(f'{results_path}/res.npy', allow_pickle=True).item()[error_type]
                            
                                    for metric in tracked_metrics:
                                        res = results[metric]
                                        if metric == 'ci':
                                            ci_low = res[0]
                                            ci_high = res[1]
                                            res = [ci_low, ci_high]
                                        results_collector[error_type][output_layer][metric].append(res)
                        
                        # plot collected results.
                        # left subplot for loc error, right subplot for rot error.
                        # x-axis is sampling rate, y-axis is decoding error.
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
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
                                            alpha=0.2,
                                            color='grey',
                                        )
                                    else:
                                        if 'baseline' in metric:
                                            # no need to label baseline for each layer
                                            # we only going to label baseline when we plot
                                            # the last layer.
                                            if output_layer == output_layers[-1]:
                                                label = metric
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
                                            color = data.load_envs_dict(model_name, envs)[
                                                f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                                        
                                        # either baseline or non-baseline layer performance,
                                        # we always plot them.
                                        axes[i].plot(
                                            sampling_rates,
                                            results_collector[error_type][output_layer][metric],
                                            label=label,
                                            color=color,
                                        )
                            axes[i].set_xlabel('sampling rates')
                            axes[i].set_title(error_type)

                        sup_title = f'{envs[0]},{movement_mode},'\
                                    f'{model_name},{feature_selection},'\
                                    f'{decoding_model_name}'\
                                    f'({decoding_model_hparams}),seed{random_seed}'
                        
                        # for across layers and sampling rates, 
                        # we save the plot at the same level as layers.
                        figs_path = f'figs/{env}/{movement_mode}/{moving_trajectory}/'\
                                    f'{model_name}/{experiment}/{feature_selection}/'\
                                    f'{decoding_model_name}_{decoding_model_hparams}'
            
                        if not os.path.exists(figs_path):
                            os.makedirs(figs_path)
                        plt.legend()
                        plt.suptitle(sup_title)
                        plt.savefig(
                            f'{figs_path}/'\
                            f'decoding_across_sampling_rates_n_layers_seed{random_seed}.png')
                        plt.close()
                        logging.info(
                            f'[Saved] {figs_path}/'
                            f'decoding_across_sampling_rates_n_layers_seed{random_seed}.png')

    elif analysis == 'decoding_across_reg_strengths_n_layers':
        # fixed sampling rate, for now use 0.5;
        # averaged over seeds.
        env = envs[0]
        moving_trajectory = moving_trajectories[0]
        movement_mode = movement_modes[0]
        sampling_rate = 0.5

        # only keep the unique decoding_model names 
        # otherwise, the same name will be iterated multiple times
        # which does absolutely nothing wrong because `reg_strengths`,
        # are looped over anyway. it is just annoying that if 
        # there are multiple decoding models with the same name 
        # (different hparams), the same operation (loop thru `reg_strengths`)
        # will be run as many times as there are decoding models with the same name.
        # in the future, we should change the data structure of `decoding_model_choices`
        # so that the keys are unique decoding model names.
        unique_reg_strengths = sorted(list(set(
            [decoding_model_choice['hparams'] \
                for decoding_model_choice in decoding_model_choices]
            )
        ))
        unique_decoding_model_names = list(set(
            [decoding_model_choice['name'] \
                for decoding_model_choice in decoding_model_choices]
            )
        )

        for model_name in model_names:
            output_layers = data.load_model_layers(model_name)
            for feature_selection in feature_selections:
                for decoding_model_name in unique_decoding_model_names:
                # for decoding_model_choice in decoding_model_choices:
                    # decoding_model_name = decoding_model_choice['name']
                    # decoding_model_hparams = decoding_model_choice['hparams']

                    if \
                        (
                            'l1' in feature_selection and \
                            decoding_model_name != 'lasso_regression'
                        ) \
                        or \
                        (
                            'l2' in feature_selection and \
                            decoding_model_name != 'ridge_regression'
                        ):
                        continue

                    # collect results across dimensions
                    # from base-case results.
                    results_collector = \
                        defaultdict(                            # key - error_type
                            lambda: defaultdict(                # key - output_layer
                                lambda: defaultdict(list)       # key - metric
                            )
                        )
                    
                    for error_type in error_types:
                        for output_layer in output_layers:
                            for reg_strength in unique_reg_strengths:
                                # reg strengths would be the base dimension where 
                                # we accumulate results in a list to plot at once.
                                to_average_over_seeds = defaultdict(list)
                                for random_seed in random_seeds:
                                    results_path = \
                                        f'results/{env}/{movement_mode}/{moving_trajectory}/'\
                                        f'{model_name}/{experiment}/{feature_selection}/'\
                                        f'{decoding_model_name}_{reg_strength}/'\
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
                    # left subplot for loc error, right subplot for rot error.
                    # x-axis is sampling rate, y-axis is decoding error.
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
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
                                        unique_reg_strengths,
                                        ci_low,
                                        ci_high,
                                        alpha=0.2,
                                        color='grey',
                                    )
                                else:
                                    if 'baseline' in metric:
                                        # no need to label baseline for each layer
                                        # we only going to label baseline when we plot
                                        # the last layer.
                                        if output_layer == output_layers[-1]:
                                            label = metric
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
                                        color = data.load_envs_dict(model_name, envs)[
                                            f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                                    
                                    # either baseline or non-baseline layer performance,
                                    # we always plot them.
                                    axes[i].plot(
                                        unique_reg_strengths,
                                        results_collector[error_type][output_layer][metric],
                                        label=label,
                                        color=color,
                                    )
                        axes[i].set_xlabel('reg strengths')
                        axes[i].set_xticks(unique_reg_strengths)
                        axes[i].set_title(error_type)
                    sup_title = f'{envs[0]},{movement_mode},'\
                                f'{model_name},{feature_selection},'\
                                f'{decoding_model_name},sr{sampling_rate}'
                    
                    # for across layers and sampling rates, 
                    # we save the plot at the same level as layers.
                    figs_path = f'figs/{env}/{movement_mode}/{moving_trajectory}/'\
                                f'{model_name}/{experiment}/{feature_selection}'
                    if not os.path.exists(figs_path):
                        os.makedirs(figs_path)
                    plt.legend()
                    plt.suptitle(sup_title)
                    plt.savefig(f'{figs_path}/{analysis}_sr{sampling_rate}.png')
                    plt.close()
                    logging.info(f'[Saved] {figs_path}/{analysis}_sr{sampling_rate}.png')

    elif analysis == 'regression_weights_across_sampling_rates':

        # TODO: think about how to unify interface later.

        env = envs[0]
        moving_trajectory = moving_trajectories[0]
        movement_mode = movement_modes[0]

        for model_name in model_names:
            output_layers = data.load_model_layers(model_name)
            for output_layer in output_layers:
                for feature_selection in feature_selections:
                    for decoding_model_choice in decoding_model_choices:
                        decoding_model_name = decoding_model_choice['name']
                        decoding_model_hparams = decoding_model_choice['hparams']
                        for random_seed in random_seeds:
                            if \
                                (
                                    'l1' in feature_selection and \
                                    decoding_model_choice['name'] != 'lasso_regression'
                                ) \
                                or \
                                (
                                    'l2' in feature_selection and \
                                    decoding_model_choice['name'] != 'ridge_regression'
                                ):
                                # logging.info(
                                #     '[Skip] feature_selection and decoding_model_choice mismatch'
                                # )
                                continue

                            # collect results across dimensions
                            # from base-case results.
                            results_collector = defaultdict(lambda: defaultdict())
                            for sampling_rate in sampling_rates:
                                results_path = \
                                    f'results/{env}/{movement_mode}/{moving_trajectory}/'\
                                    f'{model_name}/{experiment}/{feature_selection}/'\
                                    f'{decoding_model_name}_{decoding_model_hparams}/'\
                                    f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
                                results = \
                                    np.load(
                                    f'{results_path}/res.npy', allow_pickle=True).item()
                                for weights_name in tracked_regression_weights:
                                    results_collector[sampling_rate][weights_name] = results[weights_name]
                            
                            # plot collected results.
                            fig = plt.figure(figsize=(20, 20))
                            gs = fig.add_gridspec(len(sampling_rates), 4)
                            subtitles = [('x', 'b'), ('y', 'r'), ('rot', 'k')]
                            for row_idx, sampling_rate in enumerate(sampling_rates):
                                # load coef and intercept of a sampling rate  
                                coef = results_collector[sampling_rate]['coef']            # (targets, features)
                                intercept = results_collector[sampling_rate]['intercept']  # (targets,)

                                # add a subplot spans 3 columns for coef
                                ax_coef = fig.add_subplot(gs[row_idx, :-1])
                                for coef_index in range(len(subtitles)):
                                    ax_coef.plot(
                                        coef[coef_index, :], alpha=0.2, 
                                        label=subtitles[coef_index][0],
                                        color=subtitles[coef_index][1]
                                    )

                                # add a subplot of 1 column for intercept
                                ax_intercept = fig.add_subplot(gs[row_idx, -1])
                                ax_intercept.plot(intercept, label='intercept')
                                # remove redundant xticks except last row
                                if row_idx < len(sampling_rates)-1:
                                    ax_coef.set_xticks([])
                                    ax_intercept.set_xticks([])
                                else:
                                    ax_coef.set_xlabel('units')
                                    ax_intercept.set_xlabel('intercepts')

                                # add sampling rate as title
                                ax_coef.set_title(f'sampling rate: {sampling_rate}')

                            ax_coef.legend()
                            ax_intercept.legend()
                            plt.tight_layout()
                            sup_title = f'{envs[0]},{movement_mode},'\
                                        f'{model_name},{feature_selection},'\
                                        f'{decoding_model_name}({decoding_model_hparams},'\
                                        f'seed{random_seed})'
                            # for coef and intercept distribution across sampling rates, 
                            # we save the plot at the same as output layer.
                            figs_path = f'figs/{env}/{movement_mode}/{moving_trajectory}/'\
                                            f'{model_name}/{experiment}/{feature_selection}/'\
                                            f'{decoding_model_name}_{decoding_model_hparams}/'\
                                            f'{output_layer}'
                            if not os.path.exists(figs_path):
                                os.makedirs(figs_path)
                            plt.legend()
                            plt.tight_layout()
                            plt.suptitle(sup_title)
                            plt.savefig(
                                f'{figs_path}/'
                                f'regression_weights_across_sampling_rates_seed{random_seed}.png'
                            )
                            plt.close()
                            logging.info(
                                f'[Saved] {figs_path}/'
                                f'regression_weights_across_sampling_rates_seed{random_seed}.png'
                            )

    elif analysis == 'regression_weights_between_targets_correlations_across_layers':
        # Across layers and seeds, analyze how within the same setting (e.g. res.npy),
        # how much does coef wrt x, y and rot correlate with each other. Then we 
        # show an aggregate plot of these three pairs of correlations across layers and seeds.
        env = envs[0]
        moving_trajectory = moving_trajectories[0]
        movement_mode = movement_modes[0]
        tracked_regression_weights = ['coef']
        tracked_correlations = ['x_y_corr', 'x_rot_corr', 'y_rot_corr']
        absolute_coef = False

        for model_name in model_names:
            output_layers = data.load_model_layers(model_name)
            for decoding_model_choice in decoding_model_choices:
                decoding_model_name = decoding_model_choice['name']
                decoding_model_hparams = decoding_model_choice['hparams']
                for feature_selection in feature_selections:
                    if \
                        (
                            'l1' in feature_selection and \
                            decoding_model_choice['name'] != 'lasso_regression'
                        ) \
                        or \
                        (
                            'l2' in feature_selection and \
                            decoding_model_choice['name'] != 'ridge_regression'
                        ):
                        continue

                    # collect results across dimensions
                    # from base-case results.
                    # key1 - output_layer
                    # key2 - correlation type
                    # key3 - avg/std
                    results_collector = \
                        defaultdict(
                        lambda: defaultdict(
                        lambda: defaultdict(list))
                    )
                    for output_layer in output_layers:
                        for sampling_rate in sampling_rates:
                            # key1 - metric, e.g. x_y_corr, x_rot_corr, y_rot_corr
                            # key2 - random_seed
                            to_average_over_seeds = defaultdict(lambda: defaultdict(list))
                            for random_seed in random_seeds:
                                results_path = \
                                    f'results/{env}/{movement_mode}/{moving_trajectory}/'\
                                    f'{model_name}/{experiment}/{feature_selection}/'\
                                    f'{decoding_model_name}_{decoding_model_hparams}/'\
                                    f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
                                results = \
                                    np.load(
                                    f'{results_path}/res.npy', allow_pickle=True).item()
                                
                                # compute correlation between x, y and rot
                                coef = results['coef']  # (targets, features)
                                if absolute_coef:
                                    coef = np.abs(coef)
                                x_coef = coef[0, :]
                                y_coef = coef[1, :]
                                rot_coef = coef[2, :]
                                x_y_corr = np.round(stats.spearmanr(x_coef, y_coef)[0], 2)
                                x_rot_corr = np.round(stats.spearmanr(x_coef, rot_coef)[0], 2)
                                y_rot_corr = np.round(stats.spearmanr(y_coef, rot_coef)[0], 2)
                                to_average_over_seeds['x_y_corr'][random_seed].append(x_y_corr)
                                to_average_over_seeds['x_rot_corr'][random_seed].append(x_rot_corr)
                                to_average_over_seeds['y_rot_corr'][random_seed].append(y_rot_corr)
                            
                            # collect averaged over seeds results for plotting.
                            for corr_type in tracked_correlations:
                                avg_corr = np.mean(list(to_average_over_seeds[corr_type].values()))
                                std_corr = np.std(list(to_average_over_seeds[corr_type].values()))
                                results_collector[output_layer][corr_type]['avg'].append(avg_corr)
                                results_collector[output_layer][corr_type]['std'].append(std_corr)
                                logging.info(f"[Collected] {output_layer},{sampling_rate},{corr_type}")
                    
                    # plotter
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    for i, corr_type in enumerate(tracked_correlations):
                        for output_layer in output_layers:
                            axes[i].plot(
                                sampling_rates,
                                results_collector[output_layer][corr_type]['avg'],
                                label=output_layer,
                                color=data.load_envs_dict(model_name, envs)[
                                    f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                            )
                            axes[i].fill_between(
                                sampling_rates,
                                np.array(results_collector[output_layer][corr_type]['avg']) - \
                                    np.array(results_collector[output_layer][corr_type]['std']),
                                np.array(results_collector[output_layer][corr_type]['avg']) + \
                                    np.array(results_collector[output_layer][corr_type]['std']),
                                alpha=0.2,
                                color=data.load_envs_dict(model_name, envs)[
                                    f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                            )
                        axes[i].set_xlabel('sampling rates')
                        axes[i].set_xticks(sampling_rates)
                        axes[i].set_title(corr_type)
                    sup_title = f'{envs[0]},{movement_mode},'\
                                f'{model_name},{feature_selection},'\
                                f'{decoding_model_name}({decoding_model_hparams})'
                    plt.legend()
                    plt.suptitle(sup_title)
                    plt.savefig(
                        f'figs/{env}/{movement_mode}/{moving_trajectory}/'\
                        f'{model_name}/{experiment}/{feature_selection}/'\
                        f'{decoding_model_name}_{decoding_model_hparams}/'\
                        f'regression_weights_between_targets_correlations_across_layers'\
                        f'_absolute_coef={absolute_coef}.png'
                    )
                    plt.close()
            
    elif analysis == 'decoding_across_lesion_ratios_n_layers':
        env = envs[0]
        moving_trajectory = moving_trajectories[0]
        movement_mode = movement_modes[0]
        sampling_rate = 0.3
        
        # get lesion info
        ref = feature_selections[-1].split('_')[1]   # coef | gridness | borderness | ...
        rank = feature_selections[-1].split('_')[3]
        lesion_setting = f'{ref}_{rank}'
        # if not lesion by chart, we must specify the task target
        # being `loc | rot | border_dist`
        if reference_experiment != 'unit_chart':
            target = feature_selections[-1].split('_')[5]
            lesion_setting = f'{ref}_{rank}_{target}'

        lesion_ratios = [
            float(feature_selection.split('_')[4]) \
                for feature_selection in feature_selections[1:]  # exclude baseline lesion=0
        ]
        lesion_ratios = [0] + lesion_ratios  # add baseline lesion=0

        for model_name in model_names:
            output_layers = data.load_model_layers(model_name)

            for decoding_model_choice in decoding_model_choices:
                decoding_model_name = decoding_model_choice['name']
                decoding_model_hparams = decoding_model_choice['hparams']

                # collect results across dimensions
                # from base-case results.
                results_collector = \
                    defaultdict(                            # key - error_type
                        lambda: defaultdict(                # key - output_layer
                            lambda: defaultdict(list)       # key - metric
                        )
                    )
                
                for error_type in error_types:
                    for output_layer in output_layers:
                        for feature_selection in feature_selections:
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
                # left subplot for loc error, right subplot for rot error.
                # x-axis is lesion ratio, y-axis is decoding error.
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
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
                                    lesion_ratios,
                                    ci_low,
                                    ci_high,
                                    alpha=0.2,
                                    color='grey',
                                )
                            else:
                                if 'baseline' in metric:
                                    # no need to label baseline for each layer
                                    # we only going to label baseline when we plot
                                    # the last layer.
                                    if output_layer == output_layers[-1]:
                                        label = metric
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
                                    lesion_ratios,
                                    results_collector[error_type][output_layer][metric],
                                    label=label,
                                    color=color,
                                )
                    axes[i].set_xlabel('lesion ratios')
                    axes[i].set_xticks(lesion_ratios)
                    axes[i].set_title(error_type)
                    axes[i].grid()
                sup_title = f'{[lesion_setting]}, {envs[0]},{movement_mode},'\
                            f'{model_name},'\
                            f'{decoding_model_name}({decoding_model_hparams})'
                # for across layers and feature selections (lesion only), 
                # we save the plot at the same level as layers.
                figs_path = f'figs/{env}/{movement_mode}/{moving_trajectory}/'\
                                f'{model_name}/{experiment}'
                if not os.path.exists(figs_path):
                    os.makedirs(figs_path)
                plt.legend()
                plt.suptitle(sup_title)
                plt.savefig(f'{figs_path}/decoding_across_lesion_ratios_n_layers_{lesion_setting}.png')
                plt.close()
                logging.info(f'[Saved] {figs_path}/decoding_across_lesion_ratios_n_layers_{lesion_setting}.png')

    elif analysis == 'coef_correlations_across_layers':
        env = envs[0]
        moving_trajectory = moving_trajectories[0]
        movement_mode = movement_modes[0]
        random_seed = random_seeds[0]
        sampling_rate = sampling_rates[0]

        for model_name in model_names:
            output_layers = data.load_model_layers(model_name)

            for decoding_model_choice in decoding_model_choices:
                decoding_model_name = decoding_model_choice['name']
                decoding_model_hparams = decoding_model_choice['hparams']

                # Ad hoc, will change if plot more models.
                # But good for sanity check now.
                fig, axes = plt.subplots(1, 1, figsize=(10, 5))

                for output_layer in output_layers:

                    for feature_selection in feature_selections:

                        results_path = \
                            f'results/{env}/{movement_mode}/{moving_trajectory}/'\
                            f'{model_name}/{experiment}/{feature_selection}/'\
                            f'{decoding_model_name}_{decoding_model_hparams}/'\
                            f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
                        
                        coef = np.load(f'{results_path}/res.npy', allow_pickle=True).item()['coef']
                        coef = np.abs(coef)
                        coef_loc = np.mean(coef[:2, :], axis=0)
                        coef_rot = coef[2, :]
                        coef = np.vstack((coef_loc, coef_rot))
                        logging.info(f'coef_loc.shape: {coef_loc.shape}')
                        logging.info(f'coef_rot.shape: {coef_rot.shape}')
                        logging.info(f'coef.shape: {coef.shape}')
                        spearmanr, p = stats.spearmanr(coef_loc, coef_rot)
                        print(
                            coef_loc[:5]
                        )  # different seeds' coef_loc are NOT the same
                
                    # plot one layer at a time
                    # x-axis layer, y-axis correlation, label=f'{output_layer}_p={p:.1f}'
                    axes.scatter(
                        output_layers.index(output_layer),
                        spearmanr,
                        label=f'{output_layer},p={p:.1f}',
                        color=data.load_envs_dict(model_name, envs)[
                            f'{envs[0]}_{movement_mode}_{model_name}_{output_layer}']['color']
                    )
                    axes.set_xlabel('layers')
                    axes.set_xticks(range(len(output_layers)))
                    axes.set_xticklabels(output_layers)
                    axes.set_ylabel('correlation')
                    axes.set_title(f'{envs[0]},{movement_mode},{model_name}')
                    axes.legend()
                plt.grid()
                plt.tight_layout()

                figs_path = \
                    f'figs/{env}/{movement_mode}/{moving_trajectory}/'\
                    f'{model_name}/{experiment}/{feature_selection}/'\
                    f'{decoding_model_name}_{decoding_model_hparams}/'\
                    f''
                if not os.path.exists(figs_path):
                    os.makedirs(figs_path)
                plt.savefig(f'{figs_path}/coef_correlations_across_layers_seed{random_seed}_sr{sampling_rate}.png')


if __name__ == '__main__':
    start_time = time.time()
    logging_level = 'info'
    if logging_level == 'info':
        logging.basicConfig(level=logging.INFO)
    elif logging_level == 'debug':
        logging.basicConfig(level=logging.DEBUG)

    # =================================================================== #
    TF_NUM_INTRAOP_THREADS = 10
    CPU_NUM_PROCESSES = 5
    envs = ['env28_r24']
    movement_modes = ['2d']
    sampling_rates = [0.3]
    random_seeds = [1234]
    model_names = ['vgg16']
    moving_trajectories = ['uniform']
    decoding_model_choices = [{'name': 'ridge_regression', 'hparams': 1.0}]
    experiment = 'loc_n_rot'
    reference_experiment = 'unit_chart'   #'loc_n_rot|border_dist|unit_chart'
    metric = 'maxvalueinclusters'
    thr = '0'                            # if metric=='coef', thr='thr', else '0'
    rank = 'top'
    target = ''                          # if metric=='coef', target='_loc|_rot', else ''
    feature_selections = [
        'l2',
        # f'l2+lesion_{metric}_{thr}_{rank}_0.1{target}',
        # f'l2+lesion_{metric}_{thr}_{rank}_0.3{target}',
        # f'l2+lesion_{metric}_{thr}_{rank}_0.5{target}',
        # f'l2+lesion_{metric}_{thr}_{rank}_0.7{target}',
    ]
    # =================================================================== #

    # multi_envs_across_dimensions_CPU(
    multi_envs_across_dimensions_GPU(
        target_func=_single_env_decoding_error,
        envs=envs,
        experiment=experiment,
        sampling_rates=sampling_rates,
        model_names=model_names,
        moving_trajectories=moving_trajectories,
        decoding_model_choices=decoding_model_choices,
        feature_selections=feature_selections,
        random_seeds=random_seeds,
        cuda_id_list=[0, 1, 2, 3, 4, 5, 6, 7],
    )

    cross_dimension_analysis(
        # analysis='decoding_across_sampling_rates_n_layers',
        # analysis='decoding_across_lesion_ratios_n_layers',
        analysis='coef_correlations_across_layers',
        envs=envs,
        movement_modes=movement_modes,
        model_names=model_names,
        moving_trajectories=moving_trajectories,
        feature_selections=feature_selections,
        sampling_rates=sampling_rates,
        decoding_model_choices=decoding_model_choices,
        random_seeds=random_seeds,
        experiment=experiment
    )

    # print time elapsed
    logging.info(f'Time elapsed: {time.time() - start_time:.2f}s')
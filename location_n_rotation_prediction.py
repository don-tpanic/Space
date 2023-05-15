import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import time
import logging
import itertools
import multiprocessing
import numpy as np
from scipy import stats
import seaborn as sns
from collections import defaultdict
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import utils
import data
import models

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

    # TODO: if there is feature selection (e.g. place-cell score)
    # Apply to train here.

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
        '[Check] per_loc_mse_loc_samples.shape=', 
        per_loc_mse_loc_samples.shape
    )
    logging.info(
        '[Check] per_loc_mse_rot_samples.shape=', 
        per_loc_mse_rot_samples.shape
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
    logging.info(f'[Check] config_version: {config_version}')

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
        logging.info(
            '[Skip] feature_selection and decoding_model_choice mismatch'
        )
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
        logging.info('[Check] base-case exists, skipping')
        return
    else:
        logging.info('[Check] base-case does not exist, continuing')
        if config['model_name'] == 'none':
            model = None
            preprocess_func = None
        else:
            model, preprocess_func = models.load_model(
                config['model_name'], config['output_layer'])

        preprocessed_data = data.load_preprocessed_data(
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
                        results_path=results_path,
                        config=config,
                        sampling_rate=sampling_rate,
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
            envs_dict = load_envs_dict(model_name, envs)
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
                                        color = load_envs_dict(model_name, envs)[
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


def load_envs_dict(model_name, envs):
    model_layers = data.load_model_layers(model_name)
    # gradient cmap in warm colors in a list
    cmaps = sns.color_palette("Reds", len(model_layers)).as_hex()[::-1]
    if len(envs) == 1:
        prefix = f'{envs[0]}'
    else:
        raise NotImplementedError
        # TODO: 
        # 1. env28 is not flexible.
        # 2. cannot work with across different envs (e.g. decorations.)

    envs_dict = {}
    for output_layer in model_layers:
        envs_dict[
            f'{prefix}_2d_{model_name}_{output_layer}'
        ] = {
            'name': f'{prefix}',
            'n_walls': 4,
            'output_layer': output_layer,
            'color': cmaps.pop(0),
        }
    return envs_dict


if __name__ == '__main__':
    start_time = time.time()
    logging_level = 'info'
    if logging_level == 'info':
        logging.basicConfig(level=logging.INFO)
    elif logging_level == 'debug':
        logging.basicConfig(level=logging.DEBUG)

    # =================================================================== #
    TF_NUM_INTRAOP_THREADS = 8
    CPU_NUM_PROCESSES = 10
    experiment = 'loc_n_rot'
    envs = ['env28_r24']
    movement_modes = ['2d']
    sampling_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    random_seeds = [42, 1234, 999]
    model_names = ['simclrv2_r50_1x_sk0', 'resnet50', 'vgg16']
    moving_trajectories = ['uniform']
    decoding_model_choices = [
        {'name': 'ridge_regression', 'hparams': 1.0},
        {'name': 'ridge_regression', 'hparams': 0.1},
        {'name': 'lasso_regression', 'hparams': 1.0},
        {'name': 'lasso_regression', 'hparams': 0.1},
    ]
    feature_selections = ['l2', 'l1']
    # =================================================================== #

    multi_envs_across_dimensions_CPU(
        target_func=_single_env_decoding_error,
        envs=envs,
        experiment=experiment,
        sampling_rates=sampling_rates,
        model_names=model_names,
        moving_trajectories=moving_trajectories,
        decoding_model_choices=decoding_model_choices,
        feature_selections=feature_selections,
        random_seeds=random_seeds,
    )

    cross_dimension_analysis(
        analysis='regression_weights_across_sampling_rates',
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
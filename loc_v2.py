import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import time
import logging
import multiprocessing
import numpy as np
from scipy import stats
from collections import defaultdict
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import utils_v2
import data
import models

"""
Experiment script
    Location and rotation prediction.
    DNN model representations are fit on a linear regression model
    to predict agent's locations and rotations.
"""

def determine_moving_trajectory(
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


def load_train_test_data(
        model,
        config,
        preprocessed_data,
        targets_true,
        moving_trajectory,
        sampling_rate,
        results_path,
        random_seed,
    ):
    model_reps = data.load_full_dataset_model_reps(
        config=config, model=model, 
        preprocessed_data=preprocessed_data,
    )

    X_train, X_test, y_train, y_test = \
        determine_moving_trajectory(
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
    print(f'X_train.shape: {X_train.shape}')
    print(f'X_test.shape: {X_test.shape}')
    return X_train, X_test, y_train, y_test


def compute_per_loc_mse_rot_samples(
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


def fit_decoding_model(
        X_train,
        X_test,
        y_train,
        y_test,
        results_path,
        config,
        sampling_rate,
        decoding_model_choice,
        random_seed
    ):
    print(f'[Check] Fitting regression model..')

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
    per_loc_mse_rot_samples = compute_per_loc_mse_rot_samples(
        y_test[:, 2:], y_pred[:, 2:], config['n_rotations'])
    
    print(
        '[Check] per_loc_mse_loc_samples.shape=', 
        per_loc_mse_loc_samples.shape
    )
    print(
        '[Check] per_loc_mse_rot_samples.shape=', 
        per_loc_mse_rot_samples.shape
    )

    mse_loc = np.mean(per_loc_mse_loc_samples)
    mse_rot = np.mean(per_loc_mse_rot_samples)
    ci_loc = stats.bootstrap((per_loc_mse_loc_samples,), np.mean).confidence_interval
    ci_rot = stats.bootstrap((per_loc_mse_rot_samples,), np.mean).confidence_interval
    print('[Check] model results done')

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
            compute_per_loc_mse_rot_samples(
                y_test[:, 2:], mid_rot, config['n_rotations']
            )
        )
    
    print('[Check] baseline 1 done')

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
            compute_per_loc_mse_rot_samples(
                y_test[:, 2:], random_rot, config['n_rotations']
            )
        )

    print('[Check] baseline 2 done')
    return mse_loc, mse_rot, ci_loc, ci_rot, \
        decoding_model.coef_, decoding_model.intercept_, \
                baseline_predict_mid_mse_loc, baseline_predict_mid_mse_rot, \
                    baseline_predict_random_mse_loc, baseline_predict_random_mse_rot


def single_env_decoding_error(
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
    print(f'[Check] config_version: {config_version}')
    config = utils_v2.load_config(config_version)
    results_path = utils_v2.load_results_path(
        config_version=config_version, 
        experiment=experiment,
        feature_selection=feature_selection,
        decoding_model_choice=decoding_model_choice,
        sampling_rate=sampling_rate,
        moving_trajectory=moving_trajectory,
        random_seed=random_seed
    )
    
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
        load_train_test_data(
            model=model,
            config=config,
            preprocessed_data=preprocessed_data,
            targets_true=targets_true,
            moving_trajectory=moving_trajectory,
            sampling_rate=sampling_rate,
            results_path=results_path,
        )
    
    mse_loc, mse_rot, ci_loc, ci_rot, coef, intercept, \
        baseline_predict_mid_mse_loc, baseline_predict_mid_mse_rot, \
            baseline_predict_random_mse_loc, baseline_predict_random_mse_rot = \
                fit_decoding_model(
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
    res['loc']['ci_low'] = ci_loc[0]
    res['loc']['ci_high'] = ci_loc[1]
    res['rot']['ci_low'] = ci_rot[0]
    res['rot']['ci_high'] = ci_rot[1]
    res['loc']['baseline_predict_mid_mse'] = baseline_predict_mid_mse_loc
    res['loc']['baseline_predict_random_mse'] = baseline_predict_random_mse_loc
    res['rot']['baseline_predict_mid_mse'] = baseline_predict_mid_mse_rot
    res['rot']['baseline_predict_random_mse'] = baseline_predict_random_mse_rot
    res['coef'] = coef
    res['intercept'] = intercept
    np.save(f'{results_path}/res.npy')


def multi_envs_across_dimensions_CPU(
        envs,
        model_names,
        moving_trajectories,
        sampling_rates,
        feature_selections,
        decoding_model_choices,
        n_processes=10,
    ):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with multiprocessing.Pool(processes=n_processes) as pool:
        for model_name in model_names:
            envs_dict = load_envs_dict(model_name, envs)
            config_versions=list(envs_dict.keys())
            for config_version in config_versions:
                for moving_trajectory in moving_trajectories:
                    for sampling_rate in sampling_rates:
                        for feature_selection in feature_selections:
                            for decoding_model_choice in decoding_model_choices:
                                res = pool.apply_async(
                                    single_env_decoding_error,
                                    args=(
                                        config_version, 
                                        moving_trajectory,
                                        sampling_rate,
                                        feature_selection,
                                        decoding_model_choice,
                                    )
                                )
        print(res.get())
        pool.close()
        pool.join()


def across_dimensions_analysis(
        analysis='across_sampling_rates_n_layers',
        envs=['env28_r24'],
        movement_mode=['2d'],
        model_names=['resnet50'],
        moving_trajectories=['uniform'],
        feature_selection=['l2'],
        decoding_model_choices=[{'name': 'ridge_regression', 'hparams': 1.0}],
    ):
    """
    There are many ways to do cross-dimension analysis. 
    Here we start with one that is most obvious. This func
    is general enough which could accommodate other ways of
    cross-dimension analyses.
            
    e.g. 
        loc and rot errors 
            Across 
                - sampling rates, layers
            Given
                - model, feature selection, decoding model
    """

    tracked_metrics = [
        'mse', 'ci_low', 'ci_high',
        'baseline_predict_mid_mse', 'baseline_predict_random_mse'
    ]

    error_types = ['loc', 'rot']

    if analysis == 'across_sampling_rates_n_layers':
        env = envs[0]
        model_name = model_names[0]
        output_layers = data.load_model_layers(model_name)
        moving_trajectory = moving_trajectories[0]
        feature_selection = feature_selections[0]
        decoding_model_choice = decoding_model_choices[0]

        results_collector = defaultdict(
                                defaultdict(
                                    defaultdict(list)))   # TODO: cleaner approach?
        for error_type in error_types:
            for output_layer in output_layers:
                for sampling_rate in sampling_rates:
                    to_average_over_seeds = defaultdict(list)
                    for random_seed in random_seeds:
                        results_path = \
                            f'results/{env}/{movement_mode}/{moving_trajectory}/'\
                            f'{model_name}/{experiment}/{feature_selection}/'\
                            f'{decoding_model_choice}/{output_layer}/sr{sampling_rate}/seed{random_seed}'
                        results = np.load(f'{results_path}/res.npy', allow_pickle=True).item()[error_type]
                        for metric in tracked_metrics:
                            to_average_over_seeds[metric].append(results[metric])
                    
                    # per metric per output layer 
                    # across sampling rates averaged over seeds
                    for metric in tracked_metrics:
                        results_collector[error_type][output_layer][metric].append(
                            np.mean(to_average_over_seeds[metric])
                        )
        
        # plot results from results_collector
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for i, error_type in enumerate(error_types):
            for output_layer in output_layers:
                for metric in tracked_metrics:
                    axes[i].plot(
                        sampling_rates,
                        results_collector[error_type][output_layer][metric],
                        label=f'{output_layer}_{metric}'
                    )
            axes[i].legend()
            axes[i].set_xlabel('sampling rates')
            axes[i].set_ylabel(error_type)



def load_envs_dict(model_name, envs):
    model_layers = data.load_model_layers(model_name)
    colors = ['k', 'y', 'g', 'cyan']
    if len(envs) == 1:
        prefix = envs[0]
    else:
        raise NotImplementedError
        # TODO: 
        # 1. env28 is not flexible.
        # 2. cannot work with across different envs (e.g. decorations.)

    envs_dict = {}
    for output_layer in model_layers:
        envs_dict[f'{prefix}_{model_name}_{output_layer}'] = {
            'name': 'env28',
            'n_walls': 4,
            'output_layer': output_layer,
            'color': colors.pop(0),
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
    TF_NUM_INTRAOP_THREADS = 10
    experiment = 'loc_n_rot'
    envs = ['env28_r24_2d']
    movement_modes = ['uniform']
    producing_results = True
    sampling_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    random_seeds = [42, 1234, 999]
    model_names = ['simclrv2_r50_1x_sk0', 'resnet50', 'vgg16']
    moving_trajectories = ['uniform']
    decoding_model_choices = [{'name': 'ridge_regression', 'hparams': 1.0}]
    feature_selections = ['l2']
    # =================================================================== #

    # print time elapsed
    logging.info(f'Time elapsed: {time.time() - start_time:.2f}s')
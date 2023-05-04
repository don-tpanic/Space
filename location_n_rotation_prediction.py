import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

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

import utils
from models import load_model
from data import load_preprocessed_data, load_decoding_targets

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
        np.random.seed(999)
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

    return X_train, X_test, y_train, y_test


def load_train_test_data(
        model,
        config,
        preprocessed_data,
        targets_true,
        moving_trajectory,
        sampling_rate,
    ):
    unity_env = config['unity_env']
    model_name = config['model_name']
    output_layer = config['output_layer']
    movement_mode = config['movement_mode']
    feature_selection = config['feature_selection']
    results_path = \
        f'results/{unity_env}/{movement_mode}/'\
        f'{model_name}/{output_layer}/{feature_selection}/'

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # use raw image input 
    if model_name == 'none':
        model_reps = preprocessed_data.reshape(preprocessed_data.shape[0], -1)
        print(f'raw image input shape: {model_reps.shape}')
    # use model output
    else:
        # (n, 4096)
        model_reps = model.predict(preprocessed_data, verbose=1)

        # NOTE: solution to OOM for early layers is to save batches to disk
        # and merge on CPU and do whatever operations come below.
        del model
        K.clear_session()
        if len(model_reps.shape) > 2:
            # when not a fc layer, we need to flatten the output dim
            # except the batch dim.
            model_reps = model_reps.reshape(model_reps.shape[0], -1)
        print(f'model_reps.shape: {model_reps.shape}')

    X_train, X_test, y_train, y_test = \
        determine_moving_trajectory(
            model_reps=model_reps,
            targets_true=targets_true,
            moving_trajectory=moving_trajectory,
            results_path=results_path,
            n_rotations=config['n_rotations'],
            sampling_rate=sampling_rate,
            env_x_min=config['env_x_min'],
            env_x_max=config['env_x_max'],
            env_y_min=config['env_y_min'],
            env_y_max=config['env_y_max'],
        )
    print(f'X_train.shape: {X_train.shape}')
    print(f'X_test.shape: {X_test.shape}')
    return X_train, X_test, y_train, y_test, results_path


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
        decoding_model_choice
    ):
    mse_loc = []
    mse_rot = []
    ci_loc = []
    ci_rot = []

    # baseline 1, where the agent just predicts
    # the centre of the room for location and 90 degrees for rotation prediction
    baseline_predict_mid_mse_loc = []
    baseline_predict_mid_mse_rot = []

    # baseline error 2, where the agent just predicts
    # location and rotation at random
    baseline_predict_random_mse_loc = []
    baseline_predict_random_mse_rot = []

    print(f'[Check] Fitting regression model..')

    # TODO: if there is feature selection (e.g. place-cell score)
    # Apply to train here.

    if decoding_model_choice == 'linear_regression':
        decoding_model = linear_model.LinearRegression()
    elif decoding_model_choice == 'ridge_regression':
        decoding_model = linear_model.Ridge()
    decoding_model.fit(X_train, y_train)
    y_pred = decoding_model.predict(X_test)

    # save each classifier's coefficients which to be
    # plotted later (save for each sampling_rate&n_components)
    coef_ = decoding_model.coef_
    intercept_ = decoding_model.intercept_
    np.save(
        os.path.join(results_path, f'coef_{sampling_rate}.npy'),
        coef_
    )
    np.save(
        os.path.join(results_path, f'intercept_{sampling_rate}.npy'),
        intercept_
    )
    print(
        f'[Check] saved coef_ and intercept_ '
    )

    # compute element-wise MSE and average and bootstrap CI
    # for location, we compute per location MSE of coordinates
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

    mse_loc.append(
        np.mean(per_loc_mse_loc_samples)
    )
    mse_rot.append(
        np.mean(per_loc_mse_rot_samples)
    )
    ci_loc.append(
        stats.bootstrap(
        (per_loc_mse_loc_samples,), np.mean).confidence_interval
    )
    ci_rot.append(
        stats.bootstrap(
        (per_loc_mse_rot_samples,), np.mean).confidence_interval
    )
    print('[Check] model results done]')

    # baseline 1, predict mid ------------------------------------
    mid_loc = np.array(
        [(config['env_x_max']+config['env_x_min']), 
        (config['env_y_max']+config['env_y_min'])]
    )
    mid_rot = np.array([config['n_rotations']//4])

    baseline_predict_mid_mse_loc.append(
        np.mean(
            np.square(y_test[:, :2] - mid_loc)
        )
    )
    baseline_predict_mid_mse_rot.append(
        np.mean(
            compute_per_loc_mse_rot_samples(
                y_test[:, 2:], mid_rot, config['n_rotations']
            )
        )
    )
    print('[Check] baseline 1 done')

    # baseline error 2, predict random ------------------------------------
    # first, we sample random locations based on bounds of the env
    np.random.seed(999)
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

    baseline_predict_random_mse_loc.append(
        np.mean(
            np.square(y_test[:, :2] - random_loc)
        )
    )
    baseline_predict_random_mse_rot.append(
        np.mean(
            compute_per_loc_mse_rot_samples(
                y_test[:, 2:], random_rot, config['n_rotations']
            )
        )
    )
    print('[Check] baseline 2 done')
    return mse_loc, mse_rot, ci_loc, ci_rot, \
                baseline_predict_mid_mse_loc, baseline_predict_mid_mse_rot, \
                    baseline_predict_random_mse_loc, baseline_predict_random_mse_rot, \
                        results_path


def single_env_decoding_error_across_sampling_rates(
        config_version, 
        moving_trajectory,
        sampling_rates,
        decoding_model_choice
    ):
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    print(f'[Check] config_version: {config_version}')
    config = utils.load_config(config_version)
    if config['model_name'] == 'none':
        model = None
        preprocess_func = None
    else:
        model, preprocess_func = load_model(
            config['model_name'], config['output_layer'])

    preprocessed_data = load_preprocessed_data(
        data_path=f"data/unity/{config['unity_env']}/{config['movement_mode']}", 
        movement_mode=config['movement_mode'],
        x_min=config['env_x_min'],
        x_max=config['env_x_max'],
        y_min=config['env_y_min'],
        y_max=config['env_y_max'],
        multiplier=config['multiplier'],
        n_rotations=config['n_rotations'],
        preprocess_func=preprocess_func,
    )

    targets_true = load_decoding_targets(
        movement_mode=config['movement_mode'],
        x_min=config['env_x_min'],
        x_max=config['env_x_max'],
        y_min=config['env_y_min'],
        y_max=config['env_y_max'],
        multiplier=config['multiplier'],
        n_rotations=config['n_rotations'],
    )

    mse_loc_across_sampling_rates = defaultdict()
    mse_rot_across_sampling_rates = defaultdict()
    ci_loc_across_sampling_rates = defaultdict()
    ci_rot_across_sampling_rates = defaultdict()
    baseline_predict_mid_mse_loc_across_sampling_rates = defaultdict()
    baseline_predict_mid_mse_rot_across_sampling_rates = defaultdict()
    baseline_predict_random_mse_loc_across_sampling_rates = defaultdict()
    baseline_predict_random_mse_rot_across_sampling_rates = defaultdict()

    for sampling_rate in sampling_rates:
        X_train, X_test, y_train, y_test, \
            results_path = load_train_test_data(
                model=model,
                config=config,
                preprocessed_data=preprocessed_data,
                targets_true=targets_true,
                moving_trajectory=moving_trajectory,
                sampling_rate=sampling_rate,
            )
        
        mse_loc, mse_rot, ci_loc, ci_rot, \
            baseline_predict_mid_mse_loc, baseline_predict_mid_mse_rot, \
                baseline_predict_random_mse_loc, baseline_predict_random_mse_rot, \
                    results_path = fit_decoding_model(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        results_path,
                        config,
                        sampling_rate,
                        decoding_model_choice
                    )

        mse_loc_across_sampling_rates[sampling_rate] = mse_loc
        mse_rot_across_sampling_rates[sampling_rate] = mse_rot
        ci_loc_across_sampling_rates[sampling_rate] = ci_loc
        ci_rot_across_sampling_rates[sampling_rate] = ci_rot

        baseline_predict_mid_mse_loc_across_sampling_rates[sampling_rate] = \
            baseline_predict_mid_mse_loc
        baseline_predict_mid_mse_rot_across_sampling_rates[sampling_rate] = \
            baseline_predict_mid_mse_rot
        baseline_predict_random_mse_loc_across_sampling_rates[sampling_rate] = \
            baseline_predict_random_mse_loc
        baseline_predict_random_mse_rot_across_sampling_rates[sampling_rate] = \
            baseline_predict_random_mse_rot
    
        np.save(
            f'{results_path}/mse_loc_' \
            f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy',
            mse_loc_across_sampling_rates
        )
        np.save(
            f'{results_path}/mse_rot_' \
            f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy',
            mse_rot_across_sampling_rates
        )
        np.save(
            f'{results_path}/ci_loc_' \
            f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy',
            ci_loc_across_sampling_rates
        )
        np.save(
            f'{results_path}/ci_rot_' \
            f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy',
            ci_rot_across_sampling_rates
        )
        
        np.save(
            f'{results_path}/baseline_predict_mid_mse_loc_' \
            f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy',
            baseline_predict_mid_mse_loc_across_sampling_rates
        )
        np.save(
            f'{results_path}/baseline_predict_mid_mse_rot_' \
            f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy',
            baseline_predict_mid_mse_rot_across_sampling_rates
        )

        np.save(
            f'{results_path}/baseline_predict_random_mse_loc_' \
            f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy',
            baseline_predict_random_mse_loc_across_sampling_rates
        )
        np.save(
            f'{results_path}/baseline_predict_random_mse_rot_' \
            f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy',
            baseline_predict_random_mse_rot_across_sampling_rates
        )
        print('[Check] saved results.')


def single_env_regression_weights_across_sampling_rates(
        config_version, 
        sampling_rates,
    ):
    """
    Plot saved regression weights that are per sampling_rates,
    created and saved by `single_env_decoding_error_across_sampling_rates`.

    Specifically, 
        this function plots the regression weights as 4 plots each
        corresponds to the coefficients (+bias) that map features to targets [x, y, rot].
        This analysis is to see differences in distribution of regression weights to 
        predicting x, y and rot as the regression weights correspond to the features 
        which are units from our model.
    """
    results_path = utils.load_results_path(config_version)   
    subtitles = [('x', 'b'), ('y', 'r'), ('rot', 'k')]

    # top-level figure creation
    # 1 figure per sampling rate across n_components
    # at each variance explained (i.e. n_components) level
    # we create a row of 4 subplots (x, y, rot for coef + bias)
    # notice, x_axis for coef subplots are different from bias subplots,
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(len(sampling_rates), 4)
    for row_idx, sampling_rate in enumerate(sampling_rates):
        # load coef and intercept of a sampling rate  
        coef = np.load(
            f'{results_path}/coef_{sampling_rate}.npy')       # (targets, features)
        intercept = np.load(
            f'{results_path}/intercept_{sampling_rate}.npy')  # (targets,)
        logging.info(f'coef.shape: {coef.shape}')
        logging.info(f'intercept.shape: {intercept.shape}')

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
    fpath = f'{results_path}/regression_weights.png'
    plt.savefig(fpath)


def multiple_envs_decoding_error_across_sampling_rates(
        envs_dict,
        n_rotations=24,
        movement_mode='2d',
        model_name='vgg16',
        output_layer='fc2',
        sampling_rates=[0.01, 0.05],
        error_types=['loc', 'rot'],
        moving_trajectory='uniform',
    ):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # The figure has 2 cols and 1 row, 
    # the first col is for location error, 
    # the second col is for rotation error.
    fig, axs = plt.subplots(1, len(error_types), figsize=(10, 5))
    for error_type_index, error_type in enumerate(error_types):
        ax = axs[error_type_index]
        ax.set_title(f'{error_type} error')
        ax.set_xlabel('sampling rate')
        if error_type_index == 0:
            ax.set_ylabel('mean squared error')

        for env_config_version in envs_dict:
            env_results_path = utils.load_results_path(env_config_version)

            # load env results
            env_mse = np.load(
                f'{env_results_path}/mse_{error_type}_' \
                f'{moving_trajectory}_{sampling_rates[0]}' \
                f'-{sampling_rates[-1]}.npy', allow_pickle=True).ravel()[0]
            env_ci = np.load(
                f'{env_results_path}/ci_{error_type}_' \
                f'{moving_trajectory}_{sampling_rates[0]}' \
                f'-{sampling_rates[-1]}.npy', allow_pickle=True).ravel()[0]
            
            env_baseline_predict_mid_mse = np.load(
                f'{env_results_path}/baseline_predict_mid_mse_{error_type}_' \
                f'{moving_trajectory}_{sampling_rates[0]}' \
                f'-{sampling_rates[-1]}.npy', allow_pickle=True).ravel()[0]
            env_baseline_predict_random_mse = np.load(
                f'{env_results_path}/baseline_predict_random_mse_{error_type}_' \
                f'{moving_trajectory}_{sampling_rates[0]}' \
                f'-{sampling_rates[-1]}.npy', allow_pickle=True).ravel()[0]

            # convert dict-collected results to list
            env_mse = [env_mse[sr] for sr in sampling_rates]
            env_ci = [env_ci[sr] for sr in sampling_rates]

            env_baseline_predict_mid_mse = \
                [env_baseline_predict_mid_mse[sr] for sr in sampling_rates]
            env_baseline_predict_random_mse = \
                [env_baseline_predict_random_mse[sr] for sr in sampling_rates]

            # plot results across sampling_rates
            ax.plot(
                sampling_rates,
                env_mse, 
                c='k',
                alpha=0.1,
            )

            # plot confidence interval
            env_ci_low =  [env_ci[i][0][0] for i in range(len(sampling_rates))]
            env_ci_high = [env_ci[i][0][1] for i in range(len(sampling_rates))]

            ax.fill_between(
                sampling_rates,
                env_ci_low,
                env_ci_high,
                color='grey',
                alpha=(1+envs_dict[env_config_version]['n_walls'])*0.2,
                label=f'n_walls: {envs_dict[env_config_version]["n_walls"]}'
            )

        ax.plot(
            sampling_rates,
            env_baseline_predict_mid_mse,
            c='r',
            label='baseline 1: predict mid'
        )

        ax.plot(
            sampling_rates,
            env_baseline_predict_random_mse,
            c='b',
            label='baseline 2: predict random'
        )

    plt.legend()
    # save multiple envs results into across_envs/ folder
    results_path = "results/across_envs"
    if not os.path.exists(results_path): 
        os.makedirs(results_path)
    plt.savefig(f'{results_path}/decoding_error_across_sampling_rates.png')


def multicuda_execute(
        target_func, 
        config_versions,
        moving_trajectory,
        sampling_rates,
        decoding_model_choice,
        cuda_id_list=[0, 1, 2, 3, 4, 5, 6, 7],
    ):
    """
    Launch multiple 
        `single_env_decoding_error_across_sampling_rates`
    to specified GPUs.
    """
    args_list = []
    for config_version in config_versions:
        single_entry = {}
        single_entry['config_version'] = config_version
        single_entry['moving_trajectory'] = moving_trajectory
        single_entry['sampling_rates'] = sampling_rates
        single_entry['decoding_model_choice'] = decoding_model_choice
        args_list.append(single_entry)

    print(args_list)
    print(len(args_list))
    utils.cuda_manager(
        target_func, args_list, cuda_id_list
    )


if __name__ == '__main__':
    import time
    start_time = time.time()
    logging_level = 'info'
    if logging_level == 'info':
        logging.basicConfig(level=logging.INFO)
    elif logging_level == 'debug':
        logging.basicConfig(level=logging.DEBUG)

    envs_dict = {
        'env28_r24_2d_vgg16_fc2': {
            'name': 'env28',
            'n_walls': 4
        },
        'env33_r24_2d_vgg16_fc2': {
            'name': 'env33',
            'n_walls': 0
        },
    }
    sampling_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    moving_trajectory = 'uniform'
    decoding_model_choice = 'ridge_regression'

    multicuda_execute(
        target_func=\
            single_env_decoding_error_across_sampling_rates,
        config_versions=list(envs_dict.keys()),
        moving_trajectory=moving_trajectory,
        sampling_rates=sampling_rates,
        decoding_model_choice=decoding_model_choice,
    )

    multiple_envs_decoding_error_across_sampling_rates(
        envs_dict=envs_dict,
        sampling_rates=sampling_rates,
        moving_trajectory=moving_trajectory,
    )

    # single_env_regression_weights_across_sampling_rates(
    #     config_version='env28_r24_2d_vgg16_fc2', 
    #     sampling_rates=sampling_rates,
    # )

    # print time elapsed
    logging.info(f'Time elapsed: {time.time() - start_time:.2f}s')
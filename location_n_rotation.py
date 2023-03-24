import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import multiprocessing
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from collections import defaultdict
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from models import load_model
from data import load_data, load_data_targets
import dimension_reduction
import utils

"""
The idea is training a mapping from 
    raw images / CNN outputs / components 
to spatial coordinates/rotations in the environment. And if it is trainable, 
we can then test if the mapping can generalise to unseen data.

e.g., if PCA is used, the test data (visual input) is first projected 
to the extracted PCs from training data and the transformed the data is 
used to predict the location/rotation given the visual input.

The baselines would be directly using raw images / CNN outputs (of unseen) 
frames to predict locations/rotations.
"""

def fit(
        model,
        config,
        preprocessed_data,
        targets_true,
        n_components,
        moving_trajectory,
        sampling_rate,
        baseline=False, 
        baseline_feature_selection=None
    ):
    """
    Given a config and n_components, fit a mappping from 
    training data to predict location.

    moving_trajectory:
        While the real data is captured by the agent moving uniformly
        on a grid in Unity, we could manipulate the split of train/test
        to imitate different moving trajectories. This could be used to 
        investigate how well the model can generalise to unseen data (
        i.e. interpolation vs extrapolation). For now we have two options
        to acquire training data:
            1. uniform: the agent moves uniformly on a grid
            2. left: the agent moves only in the left side of the grid
            3. right: the agent moves only in the right side of the grid
    
    sampling_rate:
        Determines the train/test split ratio.

    baseline: 
        Training data is either raw images or CNN outputs
        without applying dimension reduction.

    baseline_feature_selection:
        If used, this is to select a subset of features 
        to match n_components. For now, we have two options:
            1. random: randomly select n_components features
            2. maxvar: select n_components features with max variance
    """
    unity_env = config['unity_env']
    model_name = config['model_name']
    output_layer = config['output_layer']
    movement_mode = config['movement_mode']
    reduction_method = config['reduction_method']

    # for backward compatibility 
    # when there is no reduction_hparams
    try:
        reduction_hparams = config['reduction_hparams']
    except KeyError:
        reduction_hparams = None

    results_path = \
        f'results/{unity_env}/{movement_mode}/{model_name}/{output_layer}/{reduction_method}/'

    if reduction_hparams:
        for k, v in reduction_hparams.items():
            results_path += f'_{k}{v}'

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # use raw image input 
    if model_name == 'none':
        model_reps = preprocessed_data.reshape(preprocessed_data.shape[0], -1)
        print(f'raw image input shape: {model_reps.shape}')
    # use model output
    else:
        # (n, 4096)
        model_reps = model.predict(preprocessed_data)
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
                n_rotations=config['n_rotations'],
                sampling_rate=sampling_rate,
                x_min=config['x_min'],
                x_max=config['x_max'],
                y_min=config['y_min'],
                y_max=config['y_max'],
            )
    print(f'X_train.shape: {X_train.shape}', f'y_train.shape: {len(y_train)}')
    print(f'X_test.shape: {X_test.shape}', f'y_test.shape: {len(y_test)}')
    
    if not baseline:
        components, _, fitter = \
            dimension_reduction.compute_components(
                X_train, 
                reduction_method=reduction_method,
                reduction_hparams=reduction_hparams,
            )

        if reduction_method == 'pca':
            # if pca, we need to mean-center test data
            # based on mean of training data before
            # projecting test data onto PCs.
            X_train_mean = np.mean(X_train, axis=0)
            X_train = components[:, :n_components]
            Vt = fitter.components_[:n_components, :]
            X_test -= X_train_mean
            X_test = X_test @ Vt.T
        else:
            raise NotImplementedError()

    else:
        # TODO: when baseline, for now imitate the same 
        # preprocessing as done in PCA. Though
        # I do wonder if nec; and if should adjust
        # based on NMF/ICA etc.
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean

        if baseline_feature_selection == 'maxvar':
            # sample n_components columns of X based on max variance
            # and return the column indices
            maxvar_cols = np.var(X_train, axis=0).argsort()[-n_components:]
            X_train = X_train[:, maxvar_cols]
            X_test = X_test[:, maxvar_cols]
        
        elif baseline_feature_selection == 'random':
            # sample n_components columns of X randomly
            # and return the column indices
            np.random.seed(999)
            random_cols = np.random.choice(
                X_train.shape[1], 
                size=n_components, 
                replace=False
            )
            X_train = X_train[:, random_cols]
            X_test = X_test[:, random_cols]

    print(f'[Check] Fitting LinearRegression..')
    LinearRegression_model = LinearRegression()
    LinearRegression_model.fit(X_train, y_train)
    y_pred = LinearRegression_model.predict(X_test)

    # # compute MSE wrt location
    # mse_loc = mean_squared_error(y_test[:, :2], y_pred[:, :2])
    # # compute MSE wrt rotation
    # mse_rot = mean_squared_error(y_test[:, 2:], y_pred[:, 2:])

    # compute element-wise MSE and average and bootstrap CI
    # for location, we compute per location MSE of coordinates
    per_loc_mse_loc_samples = np.mean(np.square(y_test[:, :2] - y_pred[:, :2]), axis=1)
    per_loc_mse_rot_samples = np.mean(np.square(y_test[:, 2:] - y_pred[:, 2:]), axis=1)
    print('[Check] per_loc_mse_loc_samples.shape=', per_loc_mse_loc_samples.shape)
    print('[Check] per_loc_mse_rot_samples.shape=', per_loc_mse_rot_samples.shape)

    mse_loc = np.mean(per_loc_mse_loc_samples)
    mse_rot = np.mean(per_loc_mse_rot_samples)

    ci_loc = stats.bootstrap(
        (per_loc_mse_loc_samples,), np.mean).confidence_interval
    ci_rot = stats.bootstrap(
        (per_loc_mse_rot_samples,), np.mean).confidence_interval

    return mse_loc, mse_rot, ci_loc, ci_rot, \
            y_train, y_test, y_pred, results_path


def determine_moving_trajectory(
        moving_trajectory,
        n_rotations,
        sampling_rate, 
        model_reps, 
        targets_true,
        x_min,
        x_max,
        y_min,
        y_max,
    ):
    """
    While the real data is captured by the agent moving uniformly
    on a grid in Unity, we could manipulate the split of train/test
    to imitate different moving trajectories. This could be used to 
    investigate how well the model can generalise to unseen data (
    i.e. interpolation vs extrapolation).
    """
    if moving_trajectory == 'uniform':
        # X_train, X_test, y_train, y_test = \
        #     train_test_split(
        #         model_reps, targets_true, 
        #         test_size=1-sampling_rate, 
        #         random_state=999
        # )

        # make sure a sampled loc's all rotates are in train
        # so the sampling indices need to be locs of all views
        np.random.seed(999)
        train_sample_loc_indices = np.random.choice(
            model_reps.shape[0] // n_rotations,
            size=int(sampling_rate * model_reps.shape[0] // n_rotations),
        )

        # the actual sampling indices need to be adjusted
        # to include all rotations of the sampled locs
        # in other words, each index in sampled_loc_indices
        # needs to be incremented n_rotations times
        # e.g. if sampled_loc_indices = [0, 1, 2] and 
        # n_rotations = 2, then sampled_indices = [0,1, 2,3, 4,5]
        train_sample_indices = []
        for i in train_sample_loc_indices:
            train_sample_indices.extend([i*n_rotations + j for j in range(n_rotations)])

        # now we can use the sampled indices to get the train/test data
        X_train = model_reps[train_sample_indices, :]
        y_train = targets_true[train_sample_indices, :]
        X_test = np.delete(model_reps, train_sample_indices, axis=0)
        y_test = np.delete(targets_true, train_sample_indices, axis=0)
    
    elif moving_trajectory == 'left':
        # use the first `sampling_rate` of the data
        # as training data
        n_train = int(sampling_rate * model_reps.shape[0])
        X_train = model_reps[:n_train, :]
        y_train = targets_true[:n_train, :]
        X_test = model_reps[n_train:, :]
        y_test = targets_true[n_train:, :]
    
    elif moving_trajectory == 'right':
        # use the last `sampling_rate` of the data
        # as training data
        n_train = int(sampling_rate * model_reps.shape[0])
        X_train = model_reps[-n_train:, :]
        y_train = targets_true[-n_train:, :]
        X_test = model_reps[:-n_train, :]
        y_test = targets_true[:-n_train, :]

    elif moving_trajectory == 'up':
        x_axis_coords = targets_true[:, 0]
        y_axis_coords = targets_true[:, 1]
        train_sample_indices = []
        for i, x in enumerate(x_axis_coords):
            # upper half of the grid
            if x_min <= x <= x_max and 0 <= y_axis_coords[i] <= y_max:
                train_sample_indices.append(i)
        X_train = model_reps[train_sample_indices, :]
        y_train = targets_true[train_sample_indices, :]
        X_test = np.delete(model_reps, train_sample_indices, axis=0)
        y_test = np.delete(targets_true, train_sample_indices, axis=0)
    
    elif moving_trajectory == 'down':
        x_axis_coords = targets_true[:, 0]
        y_axis_coords = targets_true[:, 1]
        train_sample_indices = []
        for i, x in enumerate(x_axis_coords):
            # lower half of the grid
            if x_min <= x <= x_max and y_min <= y_axis_coords[i] <= 0:
                train_sample_indices.append(i)
        X_train = model_reps[train_sample_indices, :]
        y_train = targets_true[train_sample_indices, :]
        X_test = np.delete(model_reps, train_sample_indices, axis=0)
        y_test = np.delete(targets_true, train_sample_indices, axis=0)

    elif moving_trajectory == 'second_quadrant':
        # second quadrant bounds: 
        # x_min =< x < x_max
        # y_min =< y < y_max
        x_axis_coords = targets_true[:, 0]
        y_axis_coords = targets_true[:, 1]
        train_sample_indices = []
        for i, x in enumerate(x_axis_coords):
            if x_min <= x <= 0 and 2 <= y_axis_coords[i] <= y_max:
                train_sample_indices.append(i)
        X_train = model_reps[train_sample_indices, :]
        y_train = targets_true[train_sample_indices, :]
        X_test = np.delete(model_reps, train_sample_indices, axis=0)
        y_test = np.delete(targets_true, train_sample_indices, axis=0)
    
    elif moving_trajectory == 'second_quadrant_w_key':
        # second quadrant bounds: 
        # x_min =< x < x_max
        # y_min =< y < y_max
        x_axis_coords = targets_true[:, 0]
        y_axis_coords = targets_true[:, 1]
        train_sample_indices = []
        for i, x in enumerate(x_axis_coords):
            if x_min <= x <= -2 and 2 <= y_axis_coords[i] <= y_max:
                train_sample_indices.append(i)
            # also add key positions to training data
            # key positions are the mid ones from the other three
            # unsampled quadrants
            if (x == -2 and y_axis_coords[i] == -2) or \
                (x == 2 and y_axis_coords[i] == -2) or \
                (x == 2 and y_axis_coords[i] == 2):
                train_sample_indices.append(i)

        X_train = model_reps[train_sample_indices, :]
        y_train = targets_true[train_sample_indices, :]
        X_test = np.delete(model_reps, train_sample_indices, axis=0)
        y_test = np.delete(targets_true, train_sample_indices, axis=0)
    
    elif moving_trajectory == 'four_quadrants':
        x_axis_coords = targets_true[:, 0]
        y_axis_coords = targets_true[:, 1]
        train_sample_indices = []
        for i, x in enumerate(x_axis_coords):
            if (x_min <= x <= -3 and 3 <= y_axis_coords[i] <= y_max) or \
                (x_min <= x <= -3 and y_min <= y_axis_coords[i] <= -3) or \
                (3 <= x <= x_max and y_min <= y_axis_coords[i] <= -3) or \
                (3 <= x <= x_max and 3 <= y_axis_coords[i] <= y_max):
                train_sample_indices.append(i)
        X_train = model_reps[train_sample_indices, :]
        y_train = targets_true[train_sample_indices, :]
        X_test = np.delete(model_reps, train_sample_indices, axis=0)
        y_test = np.delete(targets_true, train_sample_indices, axis=0)
    
    elif moving_trajectory == 'four_corners':
        # extreme case where we only sample each corner
        x_axis_coords = targets_true[:, 0]
        y_axis_coords = targets_true[:, 1]
        train_sample_indices = []
        for i, x in enumerate(x_axis_coords):
            if (x_min <= x <= x_min and y_max <= y_axis_coords[i] <= y_max) or \
                (x_min <= x <= x_min and y_min <= y_axis_coords[i] <= y_min) or \
                (x_max <= x <= x_max and y_min <= y_axis_coords[i] <= y_min) or \
                (x_max <= x <= x_max and y_max <= y_axis_coords[i] <= y_max):
                train_sample_indices.append(i)
        X_train = model_reps[train_sample_indices, :]
        y_train = targets_true[train_sample_indices, :]
        X_test = np.delete(model_reps, train_sample_indices, axis=0)
        y_test = np.delete(targets_true, train_sample_indices, axis=0)
    
    elif moving_trajectory == 'diag_quadrants':
        x_axis_coords = targets_true[:, 0]
        y_axis_coords = targets_true[:, 1]
        train_sample_indices = []
        for i, x in enumerate(x_axis_coords):
            if (x_min <= x <= -3 and 3 <= y_axis_coords[i] <= y_max) or \
                (3 <= x <= x_max and y_min <= y_axis_coords[i] <= -3):
                train_sample_indices.append(i)
        X_train = model_reps[train_sample_indices, :]
        y_train = targets_true[train_sample_indices, :]
        X_test = np.delete(model_reps, train_sample_indices, axis=0)
        y_test = np.delete(targets_true, train_sample_indices, axis=0)

    elif moving_trajectory == 'center_quadrant':
        x_axis_coords = targets_true[:, 0]
        y_axis_coords = targets_true[:, 1]
        train_sample_indices = []
        for i, x in enumerate(x_axis_coords):
            if (-2 <= x <= 2 and -2 <= y_axis_coords[i] <= 2):
                train_sample_indices.append(i)

        X_train = model_reps[train_sample_indices, :]
        y_train = targets_true[train_sample_indices, :]
        X_test = np.delete(model_reps, train_sample_indices, axis=0)
        y_test = np.delete(targets_true, train_sample_indices, axis=0)

    elif moving_trajectory == 'center_quadrant_oneshot':
        x_axis_coords = targets_true[:, 0]
        y_axis_coords = targets_true[:, 1]
        train_sample_indices = []
        for i, x in enumerate(x_axis_coords):
            if (0 <= x <= 0 and 0 <= y_axis_coords[i] <= 0):
                train_sample_indices.append(i)

        X_train = model_reps[train_sample_indices, :]
        y_train = targets_true[train_sample_indices, :]
        X_test = np.delete(model_reps, train_sample_indices, axis=0)
        y_test = np.delete(targets_true, train_sample_indices, axis=0)

    return X_train, X_test, y_train, y_test


def average_error_n_std_per_loc(error_type, y_test, y_pred):
    """
    Due to in test set, we could have multiple samples per location,
    for viz error in 2D, we have to average the error+std per location in order 
    to have one data-point per loc.

    Impl:
        1. Store unique locations in a defaultdict 
        2. Store error per location in defaultdict in a list 
        3. Average error+std per location and return unique loc-error mapping

    Args:
        if error_type == 'loc', consider the first 2 columns of y_test
        if error_type == 'rot', consider last columns of y_test
    """
    # store unique locations in a defaultdict
    # store error per location in defaultdict in a list
    unique_locs = defaultdict(list)
    for i, loc in enumerate(y_test[:, :2]):
        if error_type == 'loc':
            error = np.linalg.norm(y_test[i, :2] - y_pred[i, :2])
        elif error_type == 'rot':
            error = np.linalg.norm(y_test[i, 2:] - y_pred[i, 2:])
        unique_locs[tuple(loc)].append(error)
    
    # average error per location and return unique loc-error mapping
    average_error_per_loc_mapping = {}
    error_std_per_loc_mapping = {}
    for loc, errors in unique_locs.items():
        average_error_per_loc_mapping[loc] = np.mean(errors)
        error_std_per_loc_mapping[loc] = np.std(errors)

    return average_error_per_loc_mapping, error_std_per_loc_mapping


def WITHIN_ENV__decoding_error_across_reps(
        config_version, 
        n_components_list,
        moving_trajectory,
        sampling_rate_list,
    ):
    """
    Given an env and a fixed n_components, trajectory,
    n_rotations, and sampling_rate, we visualize decoding 
    error of loc&rot across reps in ['none', 'random', 'maxvar', 'dim_reduce'].

    For now we plot 4 figures for loc/rot decoding error:
        1. baseline (no selection on columns)
        2. baseline (random selection on columns)
        3. baseline (maxvar selection on columns)
        4. PCA (top n_components)

    returns:
        1. heatmaps representing decoding error of held out locs;
            and training samples as landmarks.

        or 

        2. decoding errors and their variations across rots.
    """
    config = utils.load_config(config_version)
    if config['model_name'] == 'none':
        preprocess_func = None
    else:
        model, preprocess_func = load_model(
            config['model_name'], config['output_layer'])

    preprocessed_data = load_data(
        data_path=f"data/unity/{config['unity_env']}/{config['movement_mode']}", 
        movement_mode=config['movement_mode'],
        x_min=config['x_min'],
        x_max=config['x_max'],
        y_min=config['y_min'],
        y_max=config['y_max'],
        multiplier=config['multiplier'],
        n_rotations=config['n_rotations'],
        preprocess_func=preprocess_func,
    )

    targets_true = load_data_targets(
        movement_mode=config['movement_mode'],
        x_min=config['x_min'],
        x_max=config['x_max'],
        y_min=config['y_min'],
        y_max=config['y_max'],
        multiplier=config['multiplier'],
        n_rotations=config['n_rotations'],
    )

    # NOTE: little hack to make the multiproc_execute interface
    # consistent with other analyses.
    n_components = n_components_list[0]
    sampling_rate = sampling_rate_list[0]

    print(f'[Check] n_components: {n_components}')
    subplots = ['none', 'random', 'maxvar', 'dim_reduce']
    error_types = ['loc', 'rot']
    fig, ax = plt.subplots(2, len(subplots), figsize=(35, 20))

    for i in range(len(subplots)):
        subplot = subplots[i]
        if subplot != 'dim_reduce':
            baseline = True
            baseline_feature_selection = subplot
            subtitle = f'baseline, sampling={subplot}'
        else:
            baseline = False
            baseline_feature_selection = None
            subtitle = f'dim_reduce'

        mse_loc, mse_rot, ci_loc, ci_rot, \
            y_train, y_test, y_pred, results_path = fit(
                model=model,
                config=config,
                preprocessed_data=preprocessed_data,
                targets_true=targets_true,
                n_components=n_components,
                moving_trajectory=moving_trajectory,
                sampling_rate=sampling_rate,
                baseline=baseline,
                baseline_feature_selection=baseline_feature_selection
            )
        
        for j in range(len(error_types)):
            error_type = error_types[j]
            if error_type == 'loc':
                mse = mse_loc
            elif error_type == 'rot':
                mse = mse_rot

            average_error_per_loc_mapping, error_std_per_loc_mapping = \
                average_error_n_std_per_loc(error_type, y_test, y_pred)

            plot_landmark_n_test_error_heatmap(
                train_coords_true=y_train[:, :2],
                average_error_per_loc_mapping=average_error_per_loc_mapping,
                env_x_min=config['x_min'],
                env_x_max=config['x_max'],
                env_y_min=config['y_min'],
                env_y_max=config['y_max'],
                ax=ax[j, i],
                title=f'{subtitle}, mse_{error_type}={mse:.2f}',
            )
    title = f'prediction_baseline_vs_components_{n_components}_{moving_trajectory}{sampling_rate}_heatmap'
    plt.suptitle(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_path}/{title}.png')


def WITHIN_ENV__decoding_error_across_reps_n_components(
        config_version, 
        n_components_list, 
        moving_trajectory,
        sampling_rate_list,
        reps=['maxvar', 'dim_reduce'],
    ):
    """
    Given an env with fixed moving trajectory and n_rotations, 
    we visualize how decoding error of loc&rot change across 
    a range of n_components and sampling rates and across 
    reps in ['none', 'random', 'maxvar', 'dim_reduce'].

    returns:
        1. saves dictionaries of mse_loc and mse_rot across n_components
        and sampling rates. E.g., a resulting dict has structure: 
        {sampling_rate1: [mse_loc1, mse_loc2, ...], ...} where in the list
        are decoding errors across `n_components_list`.

        2. plot decoding errors across n_components and sampling rates.
    
    The saved dictionary results are later used by
        `ACROSS_ENVS__decoding_error_across_reps_n_components`
    where we compare decoding errors across envs.
    """
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

    preprocessed_data = load_data(
        data_path=f"data/unity/{config['unity_env']}/{config['movement_mode']}", 
        movement_mode=config['movement_mode'],
        x_min=config['x_min'],
        x_max=config['x_max'],
        y_min=config['y_min'],
        y_max=config['y_max'],
        multiplier=config['multiplier'],
        n_rotations=config['n_rotations'],
        preprocess_func=preprocess_func,
    )

    targets_true = load_data_targets(
        movement_mode=config['movement_mode'],
        x_min=config['x_min'],
        x_max=config['x_max'],
        y_min=config['y_min'],
        y_max=config['y_max'],
        multiplier=config['multiplier'],
        n_rotations=config['n_rotations'],
    )

    fig, ax = plt.subplots(2, len(reps), figsize=(20, 5))
    for i in range(len(reps)):
        subplot = reps[i]
        print(f'[Check] subplot: {subplot}')
        if subplot != 'dim_reduce':
            baseline = True
            baseline_feature_selection = subplot
            subtitle = f'baseline_sampling={subplot}'
        else:
            baseline = False
            baseline_feature_selection = None
            subtitle = f'dim_reduce'

        sampling_rate_mse_loc_list = defaultdict(list)  # {sampling_rate1: [mse_loc1, mse_loc2, ...], ...}
        sampling_rate_mse_rot_list = defaultdict(list)
        sampling_rate_ci_loc_list = defaultdict(list)
        sampling_rate_ci_rot_list = defaultdict(list)
        for sampling_rate in sampling_rate_list:
            for n_components in n_components_list:
                # check if n_components is greater than training data sampled
                # if so skip (inc.) the rest of n_components for this sampling rate.
                # and jump right into the next sampling rate.
                n_samples = preprocessed_data.shape[0] * sampling_rate
                if subplot == 'dim_reduce' and n_components > n_samples:
                    print(f'[Check] n_components: {n_components} > n_samples: {n_samples}, '\
                          f'skip the rest of n_components')
                    break
                    
                mse_loc, mse_rot, ci_loc, ci_rot, \
                    y_train, y_test, y_pred, results_path = fit(
                        model=model,
                        config=config,
                        preprocessed_data=preprocessed_data,
                        targets_true=targets_true,
                        n_components=n_components,
                        moving_trajectory=moving_trajectory,
                        sampling_rate=sampling_rate,
                        baseline=baseline,
                        baseline_feature_selection=baseline_feature_selection
                    )
                sampling_rate_mse_loc_list[sampling_rate].append(mse_loc)
                sampling_rate_mse_rot_list[sampling_rate].append(mse_rot)
                sampling_rate_ci_loc_list[sampling_rate].append(ci_loc)
                sampling_rate_ci_rot_list[sampling_rate].append(ci_rot)
                print(
                    f'[Check] {config_version} \n- sampling_rate: {sampling_rate}, ' \
                    f'n_components: {n_components}, mse_loc: {mse_loc:.2f}, mse_rot: {mse_rot:.2f}' \
                    f', ci_loc: {ci_loc[0], ci_loc[1]}, ci_rot: {ci_rot[0], ci_rot[1]}'
                )
        
        # same mse_loc and mse_rot for all sampling rates for all n_components;
        # save one file per subplot
        np.save(
            f'{results_path}/mse_loc_{subtitle}_' \
            f'{n_components_list[0]}-{n_components_list[-1]}_' \
            f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy', 
            sampling_rate_mse_loc_list
        )
        np.save(
            f'{results_path}/mse_rot_{subtitle}_' \
            f'{n_components_list[0]}-{n_components_list[-1]}_' \
            f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
            sampling_rate_mse_rot_list
        )
        np.save(
            f'{results_path}/ci_loc_{subtitle}_' \
            f'{n_components_list[0]}-{n_components_list[-1]}_' \
            f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
            sampling_rate_ci_loc_list
        )
        np.save(
            f'{results_path}/ci_rot_{subtitle}_' \
            f'{n_components_list[0]}-{n_components_list[-1]}_' \
            f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
            sampling_rate_ci_rot_list
        )
        print('[Check] saved results.')
        

def plot_landmark_n_test_error_heatmap(
        train_coords_true,
        average_error_per_loc_mapping,
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        ax,
        title,
    ):
    """
    Plot train true coords and test (average) error as heatmap
    """
    # Get true train coords (landmarks)
    train_coords_true = np.array(train_coords_true)
    train_coords_true_x = train_coords_true[:, 0]
    train_coords_true_y = train_coords_true[:, 1]

    # Get true test coords and test error
    test_coords_true_x = []
    test_coords_true_y = []
    test_error = []
    for loc, error in average_error_per_loc_mapping.items():
        test_coords_true_x.append(loc[0])
        test_coords_true_y.append(loc[1])
        test_error.append(error)

    # Plot test errors as heatmap
    ax.scatter(
        test_coords_true_x, test_coords_true_y, 
        c=test_error, 
        cmap='Reds', s=999
    )

    # Plot true train coords
    ax.scatter(
        train_coords_true_x, train_coords_true_y, 
        label='train true', c='g', marker='x', s=900
    )
        
    # The rest
    ax.set_xlabel('Unity x axis')
    ax.set_ylabel('Unity z axis')
    ax.set_xlim(env_x_min, env_x_max)
    ax.set_ylim(env_y_min, env_y_max)
    ax.set_title(f'{title}')
    return ax
    

def ACROSS_ENVS__decoding_error_across_reps_n_components(
        envs2walls,
        n_rotations=24,
        movement_mode='2d',
        model_name='vgg16',
        output_layer='fc2',
        reduction_method='pca',
        sampling_rates=[0.01, 0.05, 0.1, 0.3, 0.5],
        n_components_list=range(1, 50, 2),
        error_types=['loc', 'rot'],
        reps=['none', 'random', 'maxvar', 'dim_reduce'],
    ):
    """
    Given a sampling rate, compare decoding error across n_components
    across a number of different envs. This is to see if with more 
    decorated walls, the overall env's decoding error will be lower.

    Each env's decoding error across sampling rates and n_components
    are already computed and saved by 
        `WITHIN_ENV__decoding_error_across_n_components`

    So we will be loading results from those saved files, e.g.
        '../mse_loc_baseline, sampling=maxvar_2-4000_uniform_0.01-0.5.npy'

    where each file is a defaultdict whose keys are sampling rates and 
    each sampling rate corresponds to a list of errors of each n_components 
    from the `n_components_list`, which is pre-defined.

    This function will creare a figure where each raw corresponds to a sampling
    rate, and first column corresponds to decoding error of location (mse_loc),
    and second column corresponds to decoding error of rotation (mse_rot).

    Within each subplot, we plot all given envs' decoding errors across `n_components_list
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for rep in reps:
        fig, axes = plt.subplots(
            nrows=len(sampling_rates), ncols=len(error_types), figsize=(9, 9))
    
        for error_type_index, error_type in enumerate(error_types):
            for env in envs2walls:
                env_spec = f'{env}_r{n_rotations}_'\
                      f'{movement_mode}_{model_name}_'\
                      f'{output_layer}_9_{reduction_method}'
                env_results_path = utils.return_results_path(env_spec)

                if rep == 'dim_reduce': 
                    temp_title_mse = f'mse_{error_type}_{rep}_'
                    temp_title_ci = f'ci_{error_type}_{rep}_'
                else:
                    temp_title_mse = f'mse_{error_type}_baseline_sampling={rep}_'
                    temp_title_ci = f'ci_{error_type}_baseline_sampling={rep}_'
                results_path_mse = \
                    f'{env_results_path}/' \
                    f'{temp_title_mse}' \
                    f'{n_components_list[0]}-{n_components_list[-1]}_' \
                    f'uniform_{sampling_rates[0]}-{sampling_rates[-1]}.npy'
                results_path_ci = \
                    f'{env_results_path}/' \
                    f'{temp_title_ci}' \
                    f'{n_components_list[0]}-{n_components_list[-1]}_' \
                    f'uniform_{sampling_rates[0]}-{sampling_rates[-1]}.npy'
                results_mse = np.load(results_path_mse, allow_pickle=True).ravel()[0]
                results_ci = np.load(results_path_ci, allow_pickle=True).ravel()[0]

                for sampling_rate_index, sampling_rate in enumerate(sampling_rates):
                    ax = axes[sampling_rate_index, error_type_index]

                    ax.plot(
                        n_components_list[:len(results_mse[sampling_rate])], 
                        results_mse[sampling_rate], 
                        c='k',
                        alpha=0.1,
                    )

                    results_ci_low = []
                    results_ci_high = []
                    for i in range(len(results_ci[sampling_rate])):
                        results_ci_low.append(results_ci[sampling_rate][i][0])
                        results_ci_high.append(results_ci[sampling_rate][i][1])

                    ax.fill_between(
                        n_components_list[:len(results_mse[sampling_rate])], 
                        results_ci_low,
                        results_ci_high,
                        color='grey',
                        alpha=(1+envs2walls[env])*0.2,
                        label=f'{env}({envs2walls[env]}walls)',
                    )

                    ax.set_ylabel(f'{error_type}')
                    ax.set_title(f'sampling rate: {sampling_rate}')
                    if error_type == 'loc':
                        if sampling_rate == 0.01:
                            ax.set_ylim(6, 10)
                        else:
                            ax.set_ylim(-0.05, 7)
                    elif error_type == 'rot':
                        ax.set_ylim(-0.05, 55)
                    if sampling_rate_index == -1:
                        ax.set_xlabel('n_components')

        plt.legend() 
        plt.tight_layout()
        plt.suptitle(f'{rep}')
        envs_as_string = ''
        for env in envs2walls:
            envs_as_string += f'{env}'
        fpath = \
            f'results/across_envs/'\
            f'decoding_error_across_reps_n_components_'\
            f'{rep}_{model_name}_{output_layer}_'\
            f'{reduction_method}_{envs_as_string}.png'
        plt.savefig(fpath)


def multiproc_execute(
        target_func, 
        config_versions,
        n_components_list,
        moving_trajectory,
        sampling_rate_list,
        n_processes,
    ):
    """
    Launch multiple 
        `WITHIN_ENV__decoding_error_across_n_components`
    to CPU processes.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "5"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    with multiprocessing.Pool(n_processes) as pool:
        for config_version in config_versions:
            pool.apply_async(
                target_func, 
                args=(
                    config_version,
                    n_components_list,
                    moving_trajectory,
                    sampling_rate_list,
                )
            )
        pool.close()
        pool.join()
       

def multicuda_execute(
        target_func, 
        config_versions,
        n_components_list,
        moving_trajectory,
        sampling_rate_list,
        cuda_id_list,
    ):
    """
    Launch multiple 
        `WITHIN_ENV__decoding_error_across_n_components`
    to specified GPUs.
    """
    args_list = []
    for config_version in config_versions:
        single_entry = {}
        single_entry['config_version'] = config_version
        single_entry['n_components_list'] = n_components_list
        single_entry['moving_trajectory'] = moving_trajectory
        single_entry['sampling_rate_list'] = sampling_rate_list
        args_list.append(single_entry)

    print(args_list)
    print(len(args_list))
    utils.cuda_manager(
        target_func, args_list, cuda_id_list
    )


if __name__ == '__main__':
    import time
    start_time = time.time()

    # multicuda_execute(
    #     WITHIN_ENV__decoding_error_across_reps,
    #     config_versions=[
    #         f'env28_r24_2d_vgg16_fc2_9_pca',
    #         f'env29_r24_2d_vgg16_fc2_9_pca',
    #     ],
    #     n_components_list=[10],
    #     moving_trajectory='uniform',
    #     sampling_rate_list=[0.1],
    #     cuda_id_list=[6, 7],
    # )

    # multicuda_execute(
    #     WITHIN_ENV__decoding_error_across_reps_n_components,
    #     config_versions=[
    #         f'env28_r24_2d_none_raw_9_pca',
    #         f'env29_r24_2d_none_raw_9_pca',
    #         f'env30_r24_2d_none_raw_9_pca',
    #         f'env31_r24_2d_none_raw_9_pca',
    #         f'env32_r24_2d_none_raw_9_pca',
    #         f'env33_r24_2d_none_raw_9_pca',
    #     ],
    #     n_components_list=[1, 2, 5, 10, 15, 20, 30, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 4000],
    #     moving_trajectory='uniform',
    #     sampling_rate_list=[0.01, 0.05, 0.1, 0.3, 0.5],
    #     cuda_id_list=[0, 1, 2, 3, 4, 5],
    # )

    ACROSS_ENVS__decoding_error_across_reps_n_components(
        envs2walls={
            'env28': 4,
            # 'env29': 3,
            # 'env30': 2,
            # 'env31': 2,
            # 'env32': 1,
            'env33': 0,
        },
        model_name='none',
        output_layer='raw',
        reduction_method='pca',
        n_components_list=[1, 2, 5, 10, 15, 20, 30, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 4000],
        reps=['maxvar']
    )

    end_time = time.time()
    time_elapsed = (end_time - start_time) / 3600
    print(f'Time elapsed: {time_elapsed} hrs')
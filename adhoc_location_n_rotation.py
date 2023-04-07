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
"""

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


def accumulate_sum(array):
    accu_array = array.copy()
    for i in range(1, len(array)):
        prev = accu_array[i-1]
        now = accu_array[i]
        accu_array[i] = prev + now
    return accu_array


def loading_train_test_data(
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
    reduction_method = config['reduction_method']

    # for backward compatibility 
    # when there is no reduction_hparams
    try:
        reduction_hparams = config['reduction_hparams']
    except KeyError:
        reduction_hparams = None

    results_path = \
        f'results/{unity_env}/{movement_mode}/'\
        f'{model_name}/{output_layer}/{reduction_method}/'

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
        model_reps = model.predict(preprocessed_data, verbose=1)
        # TODO: solution to OOM for early layers is to save batches to disk
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
                n_rotations=config['n_rotations'],
                sampling_rate=sampling_rate,
                x_min=config['x_min'],
                x_max=config['x_max'],
                y_min=config['y_min'],
                y_max=config['y_max'],
            )
    print(f'X_train.shape: {X_train.shape}', f'y_train.shape: {len(y_train)}')
    print(f'X_test.shape: {X_test.shape}', f'y_test.shape: {len(y_test)}')
    return X_train, X_test, y_train, y_test, \
            results_path, reduction_hparams


def fit(
        X_train,
        X_test,
        y_train,
        y_test,
        results_path,
        reduction_hparams,
        config,
        sampling_rate,
        variance_explained_list,
        variance_explained_interval,
    ):
    """
    # TODO: docstring
    """    
    # find n_components that explain
    # a certain amount of variance
    mse_loc_across_n_components = []
    mse_rot_across_n_components = []
    ci_loc_across_n_components = []
    ci_rot_across_n_components = []

    # baseline 1, where the agent just predicts
    # the centre of the room for location and 90 degrees for rotation prediction
    baseline_predict_mid_mse_loc_across_n_components = []
    baseline_predict_mid_mse_rot_across_n_components = []

    # baseline error 2, where the agent just predicts
    # location and rotation at random
    baseline_predict_random_mse_loc_across_n_components = []
    baseline_predict_random_mse_rot_across_n_components = []

    # into the actions.. 
    components, explained_variance_ratio, fitter = \
        dimension_reduction.compute_components(
            X_train, 
            reduction_method=config['reduction_method'],
            reduction_hparams=reduction_hparams,
        )
    # if pca, we need to mean-center test data
    # based on mean of training data before
    # projecting test data onto PCs.
    X_train_mean = np.mean(X_train, axis=0)
    X_test = X_test - X_train_mean

    # 1. This analysis we evaluate 
    # decoding error at different levels 
    # of accumulative variance explained.
    if variance_explained_interval is None:
        accumulative_variance = accumulate_sum(
            explained_variance_ratio)

        # find the number of components that explain
        # a certain amount of variance and keep the comp
        # indices in a list
        n_components_list = []
        for v in variance_explained_list:
            n_components = 0
            while accumulative_variance[n_components] < v:
                n_components += 1
            print(
                f'[Check] n_components: {n_components+1}, '\
                f'explained '\
                f'{accumulative_variance[n_components]:.2f} variance.'
            )
            n_components_list.append(n_components)

            X_train_proj = components[:, :n_components+1]
            Vt = fitter.components_[:n_components+1, :]
            X_test_proj = X_test @ Vt.T

            print(f'[Check] Fitting LinearRegression..')
            LinearRegression_model = LinearRegression()
            LinearRegression_model.fit(X_train_proj, y_train)
            y_pred = LinearRegression_model.predict(X_test_proj)

            # save each classifier's coefficients which to be
            # plotted later (save for each sampling_rate&n_components)
            coef_ = LinearRegression_model.coef_
            intercept_ = LinearRegression_model.intercept_
            np.save(
                os.path.join(results_path, f'coef_{sampling_rate}_{v}.npy'),
                coef_
            )
            np.save(
                os.path.join(results_path, f'intercept_{sampling_rate}_{v}.npy'),
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

            mse_loc = np.mean(per_loc_mse_loc_samples)
            mse_rot = np.mean(per_loc_mse_rot_samples)
            mse_loc_across_n_components.append(mse_loc)
            mse_rot_across_n_components.append(mse_rot)

            ci_loc = stats.bootstrap(
                (per_loc_mse_loc_samples,), np.mean).confidence_interval
            ci_rot = stats.bootstrap(
                (per_loc_mse_rot_samples,), np.mean).confidence_interval
            ci_loc_across_n_components.append(ci_loc)
            ci_rot_across_n_components.append(ci_rot)
            print('done1')

            # baseline 1, predict mid
            mid_loc = np.array(
                [(config['x_max']+config['x_min']), 
                (config['y_max']+config['y_min'])]
            )
            mid_rot = np.array([config['n_rotations']//4])
            baseline_predict_mid_mse_loc = np.mean(
                np.square(y_test[:, :2] - mid_loc), axis=1
            )
            baseline_predict_mid_mse_rot = compute_per_loc_mse_rot_samples(
                y_test[:, 2:], mid_rot, config['n_rotations']
            )
            print(
                '[Check] baseline_predict_mid_mse_loc.shape=', 
                baseline_predict_mid_mse_loc.shape
            )
            print(
                '[Check] baseline_predict_mid_mse_rot.shape=', 
                baseline_predict_mid_mse_rot.shape
            )
            baseline_predict_mid_mse_loc_across_n_components.append(
                np.mean(baseline_predict_mid_mse_loc)
            )
            baseline_predict_mid_mse_rot_across_n_components.append(
                np.mean(baseline_predict_mid_mse_rot)
            )
            print('done2')

            # baseline error 2, predict random
            # first, we sample random locations based on bounds of the env
            np.random.seed(999)
            random_loc = np.random.uniform(
                low=np.array([config['x_min'], config['y_min']]),
                high=np.array([config['x_max'], config['y_max']]),
                size=(y_test.shape[0], 2)
            )
            # second, we sample random rotations
            random_rot = np.random.randint(
                low=0, high=config['n_rotations'], size=(y_test.shape[0], 1)
            )
            # then we can compute error like before
            baseline_predict_random_mse_loc = np.mean(
                np.square(y_test[:, :2] - random_loc), axis=1
            )
            baseline_predict_random_mse_rot = compute_per_loc_mse_rot_samples(
                y_test[:, 2:], random_rot, config['n_rotations']
            )
            print(
                '[Check] baseline_predict_random_mse_loc.shape=', 
                baseline_predict_random_mse_loc.shape
            )
            print(
                '[Check] baseline_predict_random_mse_rot.shape=', 
                baseline_predict_random_mse_rot.shape
            )
            baseline_predict_random_mse_loc_across_n_components.append(
                np.mean(baseline_predict_random_mse_loc)
            )
            baseline_predict_random_mse_rot_across_n_components.append(
                np.mean(baseline_predict_random_mse_rot)
            )
            print('done3')

    
    # 2. This analysis we evaluate decoding error
    # with comopnents that explain (not accumulative)
    # every variance_explained_interval% of variance.
    else:
        # find the number of components that explain every `interval`th
        # variance explained and keep the comp indices in a list
        n_components_list = []
        n_components_left = 0
        accumulated_variance_so_far = 0
        for n_components_right, v in enumerate(explained_variance_ratio):
            # if the current interval's accu variance satisfied the 
            # variance_explained_interval, then we evaluate the decoding
            # error with the range of components that explain the
            # current interval's accu variance.
            if accumulated_variance_so_far+v >= variance_explained_interval:
                n_components_list.append([n_components_left, n_components_right])
        
                X_train_proj = components[:, n_components_left:n_components_right+1]
                Vt = fitter.components_[n_components_left:n_components_right+1, :]
                X_test_proj = X_test @ Vt.T

                print(f'[Check] Fitting LinearRegression..')
                LinearRegression_model = LinearRegression()
                LinearRegression_model.fit(X_train_proj, y_train)
                y_pred = LinearRegression_model.predict(X_test_proj)

                # compute element-wise MSE and average and bootstrap CI
                # for location, we compute per location MSE of coordinates
                per_loc_mse_loc_samples = np.mean(np.square(y_test[:, :2] - y_pred[:, :2]), axis=1)
                per_loc_mse_rot_samples = np.mean(np.square(y_test[:, 2:] - y_pred[:, 2:]), axis=1)
                print('[Check] per_loc_mse_loc_samples.shape=', per_loc_mse_loc_samples.shape)
                print('[Check] per_loc_mse_rot_samples.shape=', per_loc_mse_rot_samples.shape)

                mse_loc = np.mean(per_loc_mse_loc_samples)
                mse_rot = np.mean(per_loc_mse_rot_samples)
                mse_loc_across_n_components.append(mse_loc)
                mse_rot_across_n_components.append(mse_rot)

                ci_loc = stats.bootstrap(
                    (per_loc_mse_loc_samples,), np.mean).confidence_interval
                ci_rot = stats.bootstrap(
                    (per_loc_mse_rot_samples,), np.mean).confidence_interval
                ci_loc_across_n_components.append(ci_loc)
                ci_rot_across_n_components.append(ci_rot)

                # once the current range of components done,
                # we reset the accumulated variance for 
                # the next range of components; and
                # starting from the next component
                accumulated_variance_so_far = 0
                n_components_left = n_components_right + 1
            
            # if the current interval's accu variance not satisfied the
            # variance_explained_interval, then we keep accumulating
            # the variance and move on to the next component
            else:
                accumulated_variance_so_far += v

    return n_components_list, \
        mse_loc_across_n_components, mse_rot_across_n_components, \
        ci_loc_across_n_components, ci_rot_across_n_components, \
            baseline_predict_mid_mse_loc_across_n_components, baseline_predict_mid_mse_rot_across_n_components, \
            baseline_predict_random_mse_loc_across_n_components, baseline_predict_random_mse_rot_across_n_components, \
                results_path


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
    
    return X_train, X_test, y_train, y_test


def WITHIN_ENV__decoding_error_across_reps_n_components(
        config_version, 
        variance_explained_list,
        variance_explained_interval,
        moving_trajectory,
        sampling_rate_list,
        reps=['dim_reduce'],
    ):
    """
    # TODO: add docstring
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

    for i in range(len(reps)):
        subtitle = reps[i]
        sampling_rate_mse_loc_list = defaultdict(list)  # {sampling_rate1: [mse_loc1, mse_loc2, ...], ...}
        sampling_rate_mse_rot_list = defaultdict(list)
        sampling_rate_ci_loc_list = defaultdict(list)
        sampling_rate_ci_rot_list = defaultdict(list)
        sampling_rate_baseline_predict_mid_mse_loc_list = defaultdict(list)
        sampling_rate_baseline_predict_mid_mse_rot_list = defaultdict(list)
        sampling_rate_baseline_predict_random_mse_loc_list = defaultdict(list)
        sampling_rate_baseline_predict_random_mse_rot_list = defaultdict(list)
        sampling_rate_n_components_list_xticks = defaultdict(list)
        for sampling_rate in sampling_rate_list:
        
            X_train, X_test, y_train, y_test, \
                results_path, reduction_hparams = \
                    loading_train_test_data(
                        model=model,
                        config=config,
                        preprocessed_data=preprocessed_data,
                        targets_true=targets_true,
                        moving_trajectory=moving_trajectory,
                        sampling_rate=sampling_rate,
                    )
            
            n_components_list, \
                mse_loc_across_n_components, mse_rot_across_n_components, \
                ci_loc_across_n_components, ci_rot_across_n_components, \
                    baseline_predict_mid_mse_loc_across_n_components, \
                    baseline_predict_mid_mse_rot_across_n_components, \
                    baseline_predict_random_mse_loc_across_n_components, \
                    baseline_predict_random_mse_rot_across_n_components, \
                        results_path = fit(
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            results_path=results_path,
                            reduction_hparams=reduction_hparams,
                            config=config,
                            sampling_rate=sampling_rate,
                            variance_explained_list=variance_explained_list,
                            variance_explained_interval=variance_explained_interval
                        )

            sampling_rate_mse_loc_list[
                sampling_rate].extend(mse_loc_across_n_components)
            sampling_rate_mse_rot_list[
                sampling_rate].extend(mse_rot_across_n_components)
            sampling_rate_ci_loc_list[
                sampling_rate].extend(ci_loc_across_n_components)
            sampling_rate_ci_rot_list[
                sampling_rate].extend(ci_rot_across_n_components)

            # baseline 1
            sampling_rate_baseline_predict_mid_mse_loc_list[
                sampling_rate].extend(baseline_predict_mid_mse_loc_across_n_components)
            sampling_rate_baseline_predict_mid_mse_rot_list[
                sampling_rate].extend(baseline_predict_mid_mse_rot_across_n_components)

            # baseline 2
            sampling_rate_baseline_predict_random_mse_loc_list[
                sampling_rate].extend(baseline_predict_random_mse_loc_across_n_components)
            sampling_rate_baseline_predict_random_mse_rot_list[
                sampling_rate].extend(baseline_predict_random_mse_rot_across_n_components)

            sampling_rate_n_components_list_xticks[sampling_rate].extend(n_components_list)
            print(f'[Check] {config_version}, sampling_rate: {sampling_rate}')
        
        if variance_explained_interval is None:
            # same the defaultdicts named as the first and last element of the 
            # variance_explained_list and the first and last element of the
            # sampling_rate_list
            np.save(
                f'{results_path}/mse_loc_{subtitle}_' \
                f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_mse_loc_list
            )
            np.save(
                f'{results_path}/mse_rot_{subtitle}_' \
                f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_mse_rot_list
            )
            np.save(
                f'{results_path}/ci_loc_{subtitle}_' \
                f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_ci_loc_list
            )
            np.save(
                f'{results_path}/ci_rot_{subtitle}_' \
                f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_ci_rot_list
            )
            # baseline 1. predict mid
            np.save(
                f'{results_path}/baseline_predict_mid_mse_loc_{subtitle}_' \
                f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_baseline_predict_mid_mse_loc_list
            )
            np.save(
                f'{results_path}/baseline_predict_mid_mse_rot_{subtitle}_' \
                f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_baseline_predict_mid_mse_rot_list
            )
            # baseline 2. predict random
            np.save(
                f'{results_path}/baseline_predict_random_mse_loc_{subtitle}_' \
                f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_baseline_predict_random_mse_loc_list
            )
            np.save(
                f'{results_path}/baseline_predict_random_mse_rot_{subtitle}_' \
                f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_baseline_predict_random_mse_rot_list
            )
            # n_components_list_xticks
            np.save(
                f'{results_path}/n_components_list_xticks_{subtitle}_' \
                f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_n_components_list_xticks
            )
        else:
            # save the defaultdicts named as the variance_explained_interval and the
            # first and last element of the sampling_rate_list
            np.save(
                f'{results_path}/mse_loc_{subtitle}_' \
                f'var_explained_interval_{variance_explained_interval}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_mse_loc_list
            )
            np.save(
                f'{results_path}/mse_rot_{subtitle}_' \
                f'var_explained_interval_{variance_explained_interval}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_mse_rot_list
            )
            np.save(
                f'{results_path}/ci_loc_{subtitle}_' \
                f'var_explained_interval_{variance_explained_interval}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_ci_loc_list
            )
            np.save(
                f'{results_path}/ci_rot_{subtitle}_' \
                f'var_explained_interval_{variance_explained_interval}_' \
                f'{moving_trajectory}_{sampling_rate_list[0]}-{sampling_rate_list[-1]}.npy',
                sampling_rate_ci_rot_list
            )
        print('[Check] saved results.')
            

def ACROSS_ENVS__decoding_error_across_reps_n_components(
        envs2walls,
        n_rotations=24,
        movement_mode='2d',
        model_name='vgg16',
        output_layer='fc2',
        reduction_method='pca',
        sampling_rates=[0.01, 0.05, 0.1, 0.3, 0.5],
        error_types=['loc', 'rot'],
        moving_trajectory='uniform',
        reps=['dim_reduce'],
        variance_explained_list=[0.3, 0.6, 0.9],
        variance_explained_interval=None
    ):
    """
    # TODO: docstring
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for rep in reps:
        fig, axes = plt.subplots(
            nrows=len(sampling_rates), ncols=len(error_types), figsize=(18, 18))
    
        for error_type_index, error_type in enumerate(error_types):
            for env in envs2walls:
                env_spec = f'{env}_r{n_rotations}_'\
                      f'{movement_mode}_{model_name}_'\
                      f'{output_layer}_9_{reduction_method}'
                env_results_path = utils.return_results_path(env_spec)
                temp_title_mse = f'mse_{error_type}_{rep}_'
                temp_title_ci = f'ci_{error_type}_{rep}_'

                if variance_explained_interval is None:
                    results_path_mse = \
                        f'{env_results_path}/' \
                        f'{temp_title_mse}' \
                        f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                        f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy'
                    results_path_ci = \
                        f'{env_results_path}/' \
                        f'{temp_title_ci}' \
                        f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                        f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy'
                    results_path_x_ticks_components = \
                        f'{env_results_path}/' \
                        f'n_components_list_xticks_{rep}_' \
                        f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                        f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy'
                    

                    # baseline 1. predict mid
                    results_path_baseline_mse = \
                        f'{env_results_path}/baseline_predict_mid_{temp_title_mse}' \
                        f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                        f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy'
                    
                    # baseline 2. predict random
                    results_path_baseline_predict_random_mse = \
                        f'{env_results_path}/baseline_predict_random_{temp_title_mse}' \
                        f'accu_var_explained_{variance_explained_list[0]}-{variance_explained_list[-1]}_' \
                        f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy'
                    
                    # x axis are just the set of variance explained levels.
                    x_axis = np.around(variance_explained_list, 2)
                    # we also plot next to the variance the actual number of compoenents
                    # which *at least* explain that much variance
                    x_ticks_n_components = np.load(
                        results_path_x_ticks_components, 
                        allow_pickle=True).ravel()[0]

                else:
                    results_path_mse = \
                        f'{env_results_path}/' \
                        f'{temp_title_mse}' \
                        f'var_explained_interval_{variance_explained_interval}_' \
                        f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy'
                    results_path_ci = \
                        f'{env_results_path}/' \
                        f'{temp_title_ci}' \
                        f'var_explained_interval_{variance_explained_interval}_' \
                        f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.npy'
                    # x axis needs set for each sampling rate, as the number of intervals
                    # may differ across sampling rates (different number of components)

                # loading results
                results_mse = np.load(results_path_mse, allow_pickle=True).ravel()[0]
                results_ci = np.load(results_path_ci, allow_pickle=True).ravel()[0]
                # loading baseline results
                results_baseline_predict_mid_mse = np.load(
                    results_path_baseline_mse, 
                    allow_pickle=True).ravel()[0]
                results_baseline_predict_random_mse = np.load(
                    results_path_baseline_predict_random_mse, 
                    allow_pickle=True).ravel()[0]

                # begin plotting
                for sampling_rate_index, sampling_rate in enumerate(sampling_rates):
                    ax = axes[sampling_rate_index, error_type_index]

                    if variance_explained_interval is not None:
                        # getting how many intervals there are by looking into
                        # each sampling rate's result and create the x_axis
                        # which spans every variance_explained_interval
                        n_intervals = len(results_mse[sampling_rate])
                        x_axis = [variance_explained_interval*i for i in range(1, n_intervals+1)]

                    ax.plot(
                        x_axis,
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
                        x_axis,
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
                            ax.set_ylim(6, 14)
                            pass
                        else:
                            ax.set_ylim(-0.05, 12)
                            pass
                    elif error_type == 'rot':
                        ax.set_ylim(-0.05, 55)
                    if sampling_rate_index == -1:
                        ax.set_xlabel('variance explained')
                    if variance_explained_interval is None:
                        x_ticks_n_components_per_sampling_rate = x_ticks_n_components[sampling_rate]
                        x_ticklabels = [f'{x_axis[i]}\n({x_ticks_n_components_per_sampling_rate[i]})' for i in range(len(x_axis))]
                        ax.set_xticks(x_axis)
                        ax.set_xticklabels(x_ticklabels)
                    ax.grid(True)

                    # plot baselines
                    ax.plot(
                        x_axis,
                        results_baseline_predict_mid_mse[sampling_rate],
                        c='r',
                        label='baseline 1: predict mid',
                    )
                    ax.plot(
                        x_axis,
                        results_baseline_predict_random_mse[sampling_rate],
                        c='b',
                        label='baseline 2: predict random',
                    )

        plt.legend() 
        plt.tight_layout()
        plt.suptitle(f'{rep}')
        envs_as_string = ''
        for env in envs2walls:
            envs_as_string += f'{env}'
        if variance_explained_interval is None:
            fpath = \
                f'results/across_envs/'\
                f'decoding_error_across_reps_n_components_'\
                f'{rep}_{model_name}_{output_layer}_'\
                f'{reduction_method}_{envs_as_string}_acc_var_explained_'\
                f'{variance_explained_list[0]}-{variance_explained_list[-1]}_'\
                f'{moving_trajectory}_{sampling_rates[0]}-{sampling_rates[-1]}.png'
        else:
            fpath = \
                f'results/across_envs/'\
                f'decoding_error_across_reps_n_components_'\
                f'{rep}_{model_name}_{output_layer}_'\
                f'{reduction_method}_{envs_as_string}_var_explained_interval_'\
                f'{variance_explained_interval}_{moving_trajectory}_'\
                f'{sampling_rates[0]}-{sampling_rates[-1]}.png'
        plt.savefig(fpath)
       

def WITHIN_ENVS__regression_weights_across_n_components(
        config_version, 
        variance_explained_list,
        moving_trajectory,
        sampling_rate_list,
        reps=['dim_reduce'],
    ):
    """
    Plot saved regression weights that are per sampling_rate&n_components,
    created and saved by `WITHIN_ENV__decoding_error_across_reps_n_components`.

    Specifically, this function plots the regression weights as 4 histograms each
    corresponds to the coefficients (+bias) that map features to targets [x, y, rot].

    This analysis is to see differences in distribution of regression weights to 
    predicting x, y and rot as the regression weights correspond to the features 
    which are PCs or units from our model.
    """
    config = utils.load_config(config_version)
    results_path = utils.return_results_path(config_version)   
    subtitles = ['x', 'y', 'rot']

    for sampling_rate in sampling_rate_list[-2:]:
        # top-level figure creation
        # 1 figure per sampling rate across n_components
        # at each variance explained (i.e. n_components) level
        # we create a row of 6 subplots (x, y, rot for coef and bias)
        fig, axes = plt.subplots(
            len(variance_explained_list[-5:]), 6, figsize=(15, 15))
        fig.suptitle(f'sampling rate: {sampling_rate}')

        for row_idx, v in enumerate(variance_explained_list[::5]):
            # load coef and intercept of a 
            # sampling rate & n_components     
            coef = np.load(
                f'{results_path}/coef_{sampling_rate}_{v}.npy')
            intercept = np.load(
                f'{results_path}/intercept_{sampling_rate}_{v}.npy')

            # plot each coef and intercept as histograom in a subplot
            # 6 subplots in total in 1 row, 
            # columns are x, y, rot for coef and bias
            for col_idx, (c, i) in enumerate(zip(coef, intercept)):
                ax = axes[row_idx, col_idx]
                # ax.hist(c, bins=20)
                ax.plot(c)
                # ax.set_xlim(-0.7, 0.7)
                ax = axes[row_idx, col_idx+3]
                # ax.hist(i, bins=20)
                ax.plot(i)
                axes[0, col_idx].set_title(f'coef {subtitles[col_idx]}')
                axes[0, col_idx+3].set_title(f'intercept {subtitles[col_idx]}')
            axes[row_idx, 0].set_ylabel(f'v.explained: {v:.2f}')
        
        # save figure
        fpath = f'{results_path}/regression_weights_{sampling_rate}.png'
        plt.savefig(fpath)


def multicuda_execute(
        target_func, 
        config_versions,
        variance_explained_list,
        variance_explained_interval,
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
        single_entry['variance_explained_list'] = variance_explained_list
        single_entry['variance_explained_interval'] = variance_explained_interval
        single_entry['moving_trajectory'] = moving_trajectory
        single_entry['sampling_rate_list'] = sampling_rate_list
        args_list.append(single_entry)

    print(args_list)
    print(len(args_list))
    utils.cuda_manager(
        target_func, args_list, cuda_id_list
    )


def multiproc_execute(
        target_func, 
        config_versions,
        variance_explained_list,
        variance_explained_interval,
        moving_trajectory,
        sampling_rate_list,
        n_processes,
    ):
    """
    Launch multiple 
        `WITHIN_ENV__decoding_error_across_n_components`
    to CPUs
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # use apply_async
    with multiprocessing.Pool(processes=n_processes) as pool:
        for config_version in config_versions:
            results = pool.apply_async(
                target_func,
                args=(
                    config_version,
                    variance_explained_list,
                    variance_explained_interval,
                    moving_trajectory,
                    sampling_rate_list,
                )
            )
        print(results.get())
        pool.close()
        pool.join()


if __name__ == '__main__':
    import time
    start_time = time.time()

    # multiproc_execute(
    multicuda_execute(
        WITHIN_ENV__decoding_error_across_reps_n_components,
        config_versions=[
            # f'env28_r24_2d_vgg16_fc2_9_pca',
            f'env33_r24_2d_vgg16_fc2_9_pca',
        ],
        variance_explained_list=np.linspace(0.5, 0.99, 21),
        # variance_explained_list=None,
        # variance_explained_interval=0.1,
        variance_explained_interval=None,
        moving_trajectory='uniform',
        sampling_rate_list=[0.01, 0.05, 0.1, 0.3, 0.5],
        cuda_id_list=[0, 1, 2, 3, 4],
        # n_processes=40,
    )

    # ACROSS_ENVS__decoding_error_across_reps_n_components(
    #     envs2walls={
    #         'env28': 4,
    #         # 'env29': 3,
    #         # 'env30': 2,
    #         # 'env31': 2,
    #         # 'env32': 1,
    #         'env33': 0,
    #     },
    #     model_name='vgg16',
    #     output_layer='block3_pool',
    #     reduction_method='pca',
    #     reps=['dim_reduce'],
    #     sampling_rates=[0.01, 0.05, 0.1, 0.3, 0.5],
    #     variance_explained_list=np.linspace(0.5, 0.99, 21),
    #     # variance_explained_list=None,
    #     # variance_explained_interval=0.1,
    #     variance_explained_interval=None,
    # )

    WITHIN_ENVS__regression_weights_across_n_components(
        config_version='env33_r24_2d_vgg16_fc2_9_pca',
        variance_explained_list=np.linspace(0.5, 0.99, 21),
        moving_trajectory='uniform',
        sampling_rate_list=[0.01, 0.05, 0.1, 0.3, 0.5],
    )

    end_time = time.time()
    time_elapsed = (end_time - start_time) / 3600
    print(f'Time elapsed: {time_elapsed} hrs')
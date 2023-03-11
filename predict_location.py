import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from models import load_model
from data import load_data, load_data_targets
import dimension_reduction
import evaluations
import utils

"""
The idea is training a mapping from 
    raw images / CNN outputs / components 
to spatial coordinates in the environment. And if it is trainable, 
we can then test if the mapping can generalise to unseen data.

e.g., if PCA is used, the test data (visual input) is first projected 
to the extracted PCs from training data and the transformed the data is 
used to predict the location given the visual input.

The baselines would be directly using raw images / CNN outputs (of unseen) 
frames to predict locations.

QUESTIONs: 
    1. should it be that PCs are trained on all data or subset? I guess subset makes
        most sense but more difficult.
    2. should baseline adjust for n_components, (like randomly sample same pixels or maxvar)
        (note, a better sample should consider RGB channels)
    3. Frames are so similar, so LR should be able to learn to predict OK, esp if training 
        data is spread (i.e., interpolation)

TODOs: 
    1. how to best visualise results? plot the true and predict see how much off?
        (tho hard to see which is which's prediction..)
    3. is prediction more accurate if near landmark or nearby training point?
"""

def fit(
        config_version, 
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
    config = utils.load_config(config_version)
    unity_env = config['unity_env']
    model_name = config['model_name']
    output_layer = config['output_layer']
    # n_components = config['n_components']
    movement_mode = config['movement_mode']
    reduction_method = config['reduction_method']
    n_rotations = config['n_rotations']
    x_min = config['x_min']
    x_max = config['x_max']
    y_min = config['y_min']
    y_max = config['y_max']
    multiplier = config['multiplier']
    data_path = f"data/unity/{unity_env}/{movement_mode}"

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

    if model_name == 'none':
        preprocess_func = None
    else:
        model, preprocess_func = load_model(model_name, output_layer)

    preprocessed_data = load_data(
        data_path=data_path, 
        movement_mode=movement_mode,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        multiplier=multiplier,
        n_rotations=n_rotations,
        preprocess_func=preprocess_func,
    )

    targets_true = load_data_targets(
        movement_mode=movement_mode,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        multiplier=multiplier,
        n_rotations=n_rotations,
    )
    
    # use raw image input 
    if model_name == 'none':
        model_reps = preprocessed_data.reshape(preprocessed_data.shape[0], -1)
        print(f'raw image input shape: {model_reps.shape}')
    # use model output
    else:
        # (n, 4096)
        model_reps = model.predict(preprocessed_data)
        K.clear_session()
        del model
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
                sampling_rate=sampling_rate,
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
            NotImplementedError()

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

    print(f'X_test.shape: {X_test.shape}', f'y_test.shape: {len(y_test)}')
    LinearRegression_model = LinearRegression()
    LinearRegression_model.fit(X_train, y_train)
    y_pred = LinearRegression_model.predict(X_test)

    # compute MSE wrt location
    mse_loc = mean_squared_error(y_test[:, :2], y_pred[:, :2])
    # compute MSE wrt rotation
    mse_rot = mean_squared_error(y_test[:, 2:], y_pred[:, 2:])
    return mse_loc, mse_rot, y_test, y_pred, x_min, x_max, y_min, y_max, results_path


def determine_moving_trajectory(
        moving_trajectory, 
        sampling_rate, 
        model_reps, 
        targets_true
    ):
    """
    While the real data is captured by the agent moving uniformly
    on a grid in Unity, we could manipulate the split of train/test
    to imitate different moving trajectories. This could be used to 
    investigate how well the model can generalise to unseen data (
    i.e. interpolation vs extrapolation). For now we have two options
    to acquire training data:
        1. uniform: the agent moves uniformly on a grid
        2. left: the agent moves only in the left side of the grid
    """
    if moving_trajectory == 'uniform':
        X_train, X_test, y_train, y_test = \
            train_test_split(
                model_reps, targets_true, 
                test_size=1-sampling_rate, 
                random_state=999
        )
    
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

    else:
        # TODO: might have more sophisticated trajectories
        # in which we take consideration of how often 
        # landmarks are visited.
        NotImplementedError()

    return X_train, X_test, y_train, y_test


def eval_baseline_vs_components(
        config_version, 
        n_components, 
        moving_trajectory,
        sampling_rate,
    ):
    """
    Compare prediction accuracy using mapping
    trained using baseline (no dimension reduction)
    and mapping trained using projected components.

    For now we plot 4 figures:
        1. baseline (no selection on columns)
        2. baseline (random selection on columns)
        3. baseline (maxvar selection on columns)
        4. PCA (top n_components)
    """
    print(f'[Check] n_components: {n_components}')
    subplots = ['none', 'random', 'maxvar', 'dim_reduce']
    fig, ax = plt.subplots(2, len(subplots), figsize=(35, 20), dpi=300)

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

        mse_loc, mse_rot, y_test, y_pred, \
            env_x_min, env_x_max, env_y_min, env_y_max, \
                results_path = fit(
                    config_version, 
                    n_components=n_components,
                    moving_trajectory=moving_trajectory,
                    sampling_rate=sampling_rate,
                    baseline=baseline,
                    baseline_feature_selection=baseline_feature_selection
        )
        
        plot_true_vs_pred_loc(
            y_test[:, :2],     # true locations, exc rotation
            y_pred[:, :2],     # predicted locations, exc rotation
            env_x_min,
            env_x_max,
            env_y_min,
            env_y_max,
            ax=ax[0, i],
            title=f'{subtitle}, mse_loc={mse_loc:.2f}',
        )

        plot_true_vs_pred_rot(
            y_test[:, 2:],    # true rotation, exc location
            y_pred[:, 2:],    # predicted rotation, exc location
            y_test[:, :2],    # true locations to plot arrows against
            env_x_min,
            env_x_max,
            env_y_min,
            env_y_max,
            ax=ax[1, i],
            title=f'{subtitle}, mse_rot={mse_rot:.2f}',
        )

    title = f'prediction_baseline_vs_components_{n_components}_{moving_trajectory}{sampling_rate}'
    plt.suptitle(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_path}/{title}.png')


def eval_n_components(
        config_version, 
        n_components_list, 
        moving_trajectory,
        sampling_rate,):
    """
    Evaluate effect of the number of components to use
    training the mapping on the final prediction
    accuracy.
    """
    subplots = ['none', 'random', 'maxvar', 'dim_reduce']
    fig, ax = plt.subplots(1, len(subplots), figsize=(20, 5))

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

        mse_list = []
        for n_components in n_components_list:
            mse_loc, mse_rot, y_test, y_pred, \
                x_min, x_max, y_min, y_max, \
                    results_path = fit(
                        config_version, 
                        n_components=n_components,
                        moving_trajectory=moving_trajectory,
                        sampling_rate=sampling_rate,
                        baseline=baseline,
                        baseline_feature_selection=baseline_feature_selection
                    )
            mse_list.append(mse_loc)
            print(f'n_components: {n_components}, mse: {mse_loc:.2f}')
            
        ax[i].plot(n_components_list, mse_list)
        ax[i].set_xlabel('n_components')
        ax[i].set_ylabel('mse')
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_title(f'{subtitle}')
        ax[i].set_xticks(n_components_list)
        ax[i].set_xticklabels(n_components_list)
        # ax[i].set_yticks([0, 1, 10, 100, 1000, 10000, 100000])
        # ax[i].set_yticklabels([0, 1, 10, 100, 1000, 10000, 100000])

    title = f'prediction_n_comp_vs_mse_{n_components_list[0]}-{n_components_list[-1]}' \
            f'_{moving_trajectory}{sampling_rate}'
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{results_path}/{title}.png')
        

def plot_true_vs_pred_loc(
        coords_true, 
        coords_pred,
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        ax,
        title,
    ):
    """
    Plot the true and predicted coordinates
    """
    coords_true = np.array(coords_true)
    coords_pred = np.array(coords_pred)
    ax.scatter(coords_true[:, 0], coords_true[:, 1], label='true', c='g', alpha=0.5)
    ax.scatter(coords_pred[:, 0], coords_pred[:, 1], label='pred', c='b', alpha=0.5, marker='x')
    ax.set_xlabel('Unity x axis')
    ax.set_ylabel('Unity z axis')
    ax.set_xlim(env_x_min-1, env_x_max+1)
    ax.set_ylim(env_y_min-1, env_y_max+1)
    ax.set_title(f'{title}')
    return ax


def plot_true_vs_pred_rot(
        rot_true, 
        rot_pred, 
        loc_true,   # Used to plot the arrows onto.
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        ax,
        title,
    ):
    """
    Plot the true and predicted rotations
    """
    rot_true = np.array(rot_true)
    rot_pred = np.array(rot_pred)
    loc_true = np.array(loc_true)

    # Convert angles to radians
    angles_true = np.deg2rad(rot_true * 60)
    angles_pred = np.deg2rad(rot_pred * 60)    # TODO: not neat

    arrow_dx_true = np.cos(angles_true)
    arrow_dy_true = np.sin(angles_true)
    arrow_dx_pred = np.cos(angles_pred)
    arrow_dy_pred = np.sin(angles_pred)

    # Plot scatter plot with arrows
    ax.scatter(
        loc_true[:, 0], loc_true[:, 1], 
        label='true', c='g', alpha=0.5
    )
    ax.quiver(
        loc_true[:, 0], loc_true[:, 1], 
        arrow_dx_true, arrow_dy_true, 
        angles='xy', scale_units='xy', color='k', label='true rot'
    )
    ax.quiver(
        loc_true[:, 0], loc_true[:, 1], 
        arrow_dx_pred, arrow_dy_pred, 
        angles='xy', scale_units='xy', color='r', label='pred rot'
    )

    ax.set_xlabel('Unity x axis')
    ax.set_ylabel('Unity z axis')
    ax.set_xlim(env_x_min-1, env_x_max+1)
    ax.set_ylim(env_y_min-1, env_y_max+1)
    ax.set_title(f'{title}')
    return ax


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config_version = 'env13fixed_2d_vgg16_fc2_9_pca'
    moving_trajectory = 'uniform'
    sampling_rate = 0.9

    for n_components in [100]:
        eval_baseline_vs_components(
            config_version=config_version, 
            n_components=n_components,
            moving_trajectory=moving_trajectory,
            sampling_rate=sampling_rate,
        )

    # n_components_list = [1, 10, 100, 1000, 2000, 4000]
    # eval_n_components(
    #     config_version=config_version, 
    #     n_components_list=n_components_list,
    #     moving_trajectory=moving_trajectory,
    #     sampling_rate=sampling_rate,
    # )
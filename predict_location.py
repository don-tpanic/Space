import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from models import load_model
from data import load_data, load_coords_targets
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
    3. Frames are so similar, so LR should be able to learn to predict OK, esp if training 
        data is spread.

TODOs: 
    1. how to best visualise results? plot the true and predict see how much off?
        (tho hard to see which is which's prediction..)
    2. mse by n_components?
    3. is prediction more accurate if near landmark or nearby training point?
"""

def fit(config_version, n_components, baseline=False, baseline_sampling_method=None):
    """
    Single fit for a given config and n_components.
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

    results_path = f'results/{unity_env}/{movement_mode}/{model_name}/{output_layer}/{reduction_method}/'
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

    coords_true = load_coords_targets(
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
        if len(model_reps.shape) > 2:
            # when not a fc layer, we need to flatten the output dim
            # except the batch dim.
            model_reps = model_reps.reshape(model_reps.shape[0], -1)
        print(f'model_reps.shape: {model_reps.shape}')

    if movement_mode == '2d':
        # due to each position in 2d takes 6 views, we 
        # concat them so each position has 1 vector.
        # also we use the fact the file names are sorted so
        # every `n_rotations` files are the same position
        n_rows = int(model_reps.shape[0] / n_rotations)
        n_cols = int(model_reps.shape[1] * n_rotations)
        model_reps = model_reps.reshape((n_rows, n_cols))
        print(f'model_reps.shape: {model_reps.shape}')

    X_train, X_test, y_train, y_test = \
        train_test_split(
            model_reps, coords_true, test_size=0.1, random_state=999
        )
    print(f'X_train.shape: {X_train.shape}', f'y_train.shape: {len(y_train)}')
    print(f'X_test.shape: {X_test.shape}', f'y_test.shape: {len(y_test)}')
    
    if not baseline:
        components, _, self_ = \
            dimension_reduction.compute_components(
                X_train, 
                reduction_method=reduction_method,
                reduction_hparams=reduction_hparams,
            )
        X_train_mean = np.mean(X_train, axis=0)
        X_train = components[:, :n_components]
        Vt = self_.components_[:n_components, :]
        X_test -= X_train_mean
        X_test = X_test @ Vt.T
        print(f'X_test.shape: {X_test.shape}', f'y_test.shape: {len(y_test)}')
    
    else:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean

        # TODO: double check this tmr.
        if baseline_sampling_method == 'maxvar':
            # sample n_components columns of X based on max variance
            # and return the column indices
            maxvar_cols = np.var(X_train, axis=0).argsort()[-n_components:]
            X_train = X_train[:, maxvar_cols]
            X_test = X_test[:, maxvar_cols]
        
        elif baseline_sampling_method == 'random':
            # sample n_components columns of X randomly
            # and return the column indices
            random_cols = np.random.choice(
                X_train.shape[1], 
                size=n_components, 
                replace=False
            )
            X_train = X_train[:, random_cols]
            X_test = X_test[:, random_cols]

    LinearRegression_model = LinearRegression()
    LinearRegression_model.fit(X_train, y_train)
    y_pred = LinearRegression_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_test, y_pred, x_min, x_max, y_min, y_max, results_path


def eval_baseline_vs_components(config_version, n_components=9):
    """
    Compare prediction accuracy using mapping
    trained using baseline (no dimension reduction)
    and mapping trained using projected components.
    """
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    # plot baseline (no sampling columns)
    mse, y_test, y_pred, \
        env_x_min, env_x_max, env_y_min, env_y_max, \
            results_path = fit(
                config_version, 
                n_components=n_components, 
                baseline=True,
                baseline_sampling_method='none')
    
    plot_true_vs_pred(
        y_test, 
        y_pred, 
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        ax=ax[0],
        title=f'baseline, mse={mse:.2f}, sampling=none',
    )

    # plot baseline (random sampling columns)
    mse, y_test, y_pred, \
        env_x_min, env_x_max, env_y_min, env_y_max, \
            results_path = fit(
                config_version, 
                n_components=n_components, 
                baseline=True,
                baseline_sampling_method='random')
    
    plot_true_vs_pred(
        y_test, 
        y_pred, 
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        ax=ax[1],
        title=f'baseline, mse={mse:.2f}, sampling=random, n_comp.={n_components}',
    )

    # plot baseline (maxvar sampling columns)
    mse, y_test, y_pred, \
        x_min, x_max, y_min, y_max, \
            results_path = fit(
                config_version, 
                n_components=n_components, 
                baseline=True,
                baseline_sampling_method='maxvar'
            )
    
    plot_true_vs_pred(
        y_test, 
        y_pred, 
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        ax=ax[2],
        title=f'baseline, mse={mse:.2f}, sampling=maxvar, n_comp.={n_components}',
    )

    # plot components
    mse, y_test, y_pred, \
        x_min, x_max, y_min, y_max, \
            results_path = fit(
                config_version, 
                n_components=n_components, 
                baseline=False,
            )

    plot_true_vs_pred(
        y_test, 
        y_pred, 
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        title=f'ours, mse={mse:.2f}, n_comp.={n_components}',
        ax=ax[3]
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_path}/compare_baseline_vs_components_prediction_{n_components}.png')


def eval_n_components(config_version, n_components_list):
    """
    Evaluate the number of components to use
    training the mapping on the final prediction
    accuracy.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    mse_list = []
    for n_components in n_components_list:
        mse, y_test, y_pred, \
            x_min, x_max, y_min, y_max, \
                results_path = fit(
                    config_version, 
                    n_components=n_components, 
                    baseline=True,
                    baseline_sampling_method='maxvar'
                )
        mse_list.append(mse)
        print(f'n_components: {n_components}, mse: {mse:.2f}')
    ax[0].plot(n_components_list, mse_list)
    ax[0].set_xlabel('n_components')
    ax[0].set_ylabel('mse')
    ax[0].set_title('baseline, sampling=maxvar')

    mse_list = []
    for n_components in n_components_list:
        mse, y_test, y_pred, \
            x_min, x_max, y_min, y_max, \
                results_path = fit(
                    config_version, 
                    n_components=n_components, 
                    baseline=False,
                )
        mse_list.append(mse)
        print(f'n_components: {n_components}, mse: {mse:.2f}')
    
    # convert to log scale if too different
    if n_components_list[-1] - n_components_list[0] > 10:
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')

    ax[1].plot(n_components_list, mse_list)
    ax[1].set_xlabel('n_components')
    ax[1].set_ylabel('mse')
    ax[1].set_title('ours')

    plt.tight_layout()
    plt.savefig(f'{results_path}/n_components_vs_mse_{n_components_list[0]}-{n_components_list[-1]}.png')
        

def plot_true_vs_pred(
        coords_true, 
        coords_pred, 
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        ax,
        title,
    ):
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


if __name__ == '__main__':
    config_version = 'env8_2d_none_raw_9_pca'

    for n_components in [1, 100, 1000]:
        eval_baseline_vs_components(config_version=config_version, n_components=n_components)

    # n_components_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # n_components_list = [1, 9, 90, 900, 9000, 90000, 900000]
    # eval_n_components(config_version=config_version, n_components_list=n_components_list)
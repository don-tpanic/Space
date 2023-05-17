import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import time
import logging
import itertools
import multiprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import utils
import data
import models

"""
Visualize individual model units see 
if there are any patterns (e.g., place cells.)
"""

def _single_model_reps(config):
    """
    Produce model_reps either directly computing if the first time,
    or load from disk if already computed.

    return:
        model_reps: \in (n_locations, n_rotations, n_features)
    """
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
    logging.info(f'model_reps.shape: {model_reps.shape}')
    return model_reps


def _single_env_viz_units(
        config_version, 
        experiment,
        reference_experiment,
        feature_selection, 
        decoding_model_choice,
        sampling_rate,
        moving_trajectory,
        random_seed,
        filterings,    
    ):
    """
    Plot individual model units of each rotation independently.
    """
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{TF_NUM_INTRAOP_THREADS}"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"

    config = utils.load_config(config_version)
    reference_experiment_results_path = \
            utils.load_results_path(
                config=config,
                experiment=reference_experiment,  # Dirty but coef is saved in loc_n_rot
                feature_selection=feature_selection,
                decoding_model_choice=decoding_model_choice,
                sampling_rate=sampling_rate,
                moving_trajectory=moving_trajectory,
                random_seed=random_seed,
    )
    logging.info(
        f'Loading results (for coef) from {reference_experiment_results_path}'
    )
    if reference_experiment_results_path is None:
        logging.info(
            f'Mismatch between feature '\
            f'selection and decoding model, skip.'
        )
        return

    movement_mode=config['movement_mode']
    env_x_min=config['env_x_min']
    env_x_max=config['env_x_max']
    env_y_min=config['env_y_min']
    env_y_max=config['env_y_max']
    multiplier=config['multiplier']
    
    # load model outputs
    model_reps = _single_model_reps(config)
    
    # TODO: feature selection based on rob metric or l1/l2
    # notice, one complexity is coef is x, y, rot
    # whereas rob metric may not be differentiating (x, y)
    # one idea is to separately plot for x, y, rot 
    if feature_selection in ['l1', 'l2']:
        # load regression coefs as selection criteria
        # for model_reps (per unit)
        targets = ['x', 'y', 'rot']
        coef = \
            np.load(
                f'{reference_experiment_results_path}/res.npy', 
                allow_pickle=True).item()['coef']  # (n_targets, n_features)
        logging.info(f'Loaded coef.shape: {coef.shape}')

        for target_index in range(coef.shape[0]):
            # filter columns of `model_reps` 
            # based on each coef of each target
            # based on `n_units_filtering` and `filtering_order`
            for filtering in filterings:
                n_units_filtering = filtering['n_units_filtering']
                filtering_order = filtering['filtering_order']
                if filtering_order == 'top_n':
                    filtered_n_units_indices = np.argsort(coef[target_index, :])[::-1][:n_units_filtering]
                    model_reps_filtered = model_reps[:, :, filtered_n_units_indices]
                elif filtering_order == 'bottom_n':
                    filtered_n_units_indices = np.argsort(coef[target_index, :])[:n_units_filtering]
                    model_reps_filtered = model_reps[:, :, filtered_n_units_indices]
                coef_filtered = coef[target_index, filtered_n_units_indices]

                fig, axes = plt.subplots(
                    nrows=model_reps_filtered.shape[2], 
                    ncols=model_reps_filtered.shape[1], 
                    figsize=(25, 25)
                )
                for unit_index in range(model_reps_filtered.shape[2]):
                    for rotation in range(model_reps_filtered.shape[1]):
                        if movement_mode == '2d':
                            # reshape to (n_locations, n_rotations, n_features)
                            heatmap = model_reps_filtered[:, rotation, unit_index].reshape(
                                (env_x_max*multiplier-env_x_min*multiplier+1, 
                                env_y_max*multiplier-env_y_min*multiplier+1)
                            )

                            # rotate heatmap to match Unity coordinate system
                            # ref: tests/testReshape_forHeatMap.py
                            heatmap = np.rot90(heatmap, k=1, axes=(0, 1))

                            # plot heatmap
                            axes[unit_index, rotation].imshow(heatmap)
                            axes[-1, rotation].set_xlabel('Unity x-axis')
                            axes[unit_index, 0].set_ylabel('Unity z-axis')
                            axes[unit_index, rotation].set_xticks([])
                            axes[unit_index, rotation].set_yticks([])
                            axes[unit_index, rotation].set_title(
                                f'u{filtered_n_units_indices[unit_index]},r{rotation}')

                sup_title = f"{filtering_order},{targets[target_index]}, "\
                            f"{config['unity_env']},{movement_mode},"\
                            f"{config['model_name']},{feature_selection}"\
                            f"({decoding_model_choice['hparams']}),"\
                            f"sr{sampling_rate},seed{random_seed}"
                
                figs_path = utils.load_figs_path(
                    config=config,
                    experiment=experiment,
                    reference_experiment='loc_n_rot',
                    feature_selection=feature_selection,
                    decoding_model_choice=decoding_model_choice,
                    sampling_rate=sampling_rate,
                    moving_trajectory=moving_trajectory,
                    random_seed=random_seed,
                )
                # fix suptitle overlap.
                fig.tight_layout(rect=[0, 0.03, 1, 0.98])
                plt.suptitle(sup_title)
                plt.savefig(
                    f'{figs_path}/units_heatmaps_{targets[target_index]}'\
                    f'_{filtering_order}.png'
                )
                plt.close()
                logging.info(
                    f'[Saved] units heatmaps {targets[target_index]}'\
                    f'{filtering_order} to {figs_path}'
                )

                # plot summed over rotation heatmap and distribution of loc-wise
                # activation intensities.
                model_reps_filtered = np.sum(
                    model_reps_filtered, axis=1, keepdims=True)

                # 1 for heatmap, 1 for distribution
                fig, axes = plt.subplots(
                    nrows=model_reps_filtered.shape[2], 
                    ncols=2,
                    figsize=(5, 25)
                )
                for unit_index in range(model_reps_filtered.shape[2]):
                    for rotation in range(model_reps_filtered.shape[1]):
                        if movement_mode == '2d':
                            # reshape to (n_locations, n_rotations, n_features)
                            heatmap = model_reps_filtered[:, rotation, unit_index].reshape(
                                (env_x_max*multiplier-env_x_min*multiplier+1, 
                                env_y_max*multiplier-env_y_min*multiplier+1)
                            )

                            # rotate heatmap to match Unity coordinate system
                            # ref: tests/testReshape_forHeatMap.py
                            heatmap = np.rot90(heatmap, k=1, axes=(0, 1))

                            # plot heatmap on the left column.
                            axes[unit_index, 0].imshow(heatmap)
                            axes[-1, 0].set_xlabel('Unity x-axis')
                            axes[-1, 0].set_ylabel('Unity z-axis')
                            axes[unit_index, 0].set_xticks([])
                            axes[unit_index, 0].set_yticks([])
                            axes[unit_index, 0].set_title(
                                f'u{filtered_n_units_indices[unit_index]},'\
                                f'coef{coef_filtered[unit_index]:.2f}'
                            )
                            
                            # plot distribution on the right column.
                            axes[unit_index, 1].hist(
                                model_reps_filtered[:, rotation, unit_index],
                                bins=10,
                            )
                            axes[-1, 1].set_xlabel('Activation intensity')

                sup_title = f"{filtering_order},{targets[target_index]},"\
                            f"{config['unity_env']},{movement_mode},"\
                            f"{config['model_name']},{feature_selection}"\
                            f"({decoding_model_choice['hparams']}),"\
                            f"sr{sampling_rate},seed{random_seed}"
                
                figs_path = utils.load_figs_path(
                    config=config,
                    experiment=experiment,
                    reference_experiment='loc_n_rot',
                    feature_selection=feature_selection,
                    decoding_model_choice=decoding_model_choice,
                    sampling_rate=sampling_rate,
                    moving_trajectory=moving_trajectory,
                    random_seed=random_seed,
                )
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.suptitle(sup_title)
                plt.savefig(
                    f'{figs_path}/units_heatmaps_{targets[target_index]}_'\
                    f'{filtering_order}_summed.png')
                plt.close()
                logging.info(
                    f'[Saved] units heatmaps {targets[target_index]} {filtering_order} '\
                    f'(summed) to {figs_path}')

    else:
        # TODO: metric-based feature selection.
        raise NotImplementedError


def multi_envs_inspect_units_CPU(
        target_func,
        envs,
        model_names,
        experiment,
        reference_experiment,
        moving_trajectories,
        sampling_rates,
        feature_selections,
        decoding_model_choices,
        random_seeds,
        filterings,
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
                                            experiment,
                                            reference_experiment,
                                            feature_selection, 
                                            decoding_model_choice,
                                            sampling_rate,
                                            moving_trajectory,
                                            random_seed,
                                            filterings,
                                        )
                                    )
        logging.info(res.get())
        pool.close()
        pool.join()


def multi_envs_inspect_units_GPU(
        target_func,
        envs,
        model_names,
        experiment, 
        reference_experiment,
        moving_trajectories,
        sampling_rates,
        feature_selections,
        decoding_model_choices,
        random_seeds,
        filterings,
        cuda_id_list=[0, 1, 2, 3, 4, 5, 6, 7],
    ):
    for model_name in model_names:
        envs_dict = load_envs_dict(model_name, envs)
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
                                single_entry['reference_experiment'] = reference_experiment
                                single_entry['feature_selection'] = feature_selection
                                single_entry['decoding_model_choice'] = decoding_model_choice
                                single_entry['random_seed'] = random_seed
                                single_entry['filterings'] = filterings
                                args_list.append(single_entry)

        logging.info(f'args_list = {args_list}')
        logging.info(f'args_list len = {len(args_list)}')
        utils.cuda_manager(
            target_func, args_list, cuda_id_list
        )
        # TODO: is this indent correct?


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

    # ======================================== #
    TF_NUM_INTRAOP_THREADS = 10
    CPU_NUM_PROCESSES = 10         
    experiment = 'viz'
    reference_experiment = 'loc_n_rot'
    envs = ['env28_r24']
    movement_modes = ['2d']
    # sampling_rates = [0.01, 0.1, 0.5]
    sampling_rates = [0.1]
    # random_seeds = [42, 1234, 999]
    random_seeds = [42]
    # model_names = ['simclrv2_r50_1x_sk0', 'resnet50', 'vgg16']
    model_names = ['vgg16']
    moving_trajectories = ['uniform']
    decoding_model_choices = [
        {'name': 'ridge_regression', 'hparams': 1.0},
        # {'name': 'ridge_regression', 'hparams': 0.1},
        # {'name': 'lasso_regression', 'hparams': 1.0},
        # {'name': 'lasso_regression', 'hparams': 0.1},
    ]
    # feature_selections = ['l2', 'l1']
    feature_selections = ['l2']
    filterings = [
        {'filtering_order': 'top_n', 'n_units_filtering': 20},
        # {'filtering_order': 'bottom_n', 'n_units_filtering': 20},
    ]
    # ======================================== #
    
    multi_envs_inspect_units_GPU(
    # multi_envs_inspect_units_CPU(
        target_func=_single_env_viz_units,
        envs=envs,
        model_names=model_names,
        experiment=experiment,
        reference_experiment=reference_experiment,
        moving_trajectories=moving_trajectories,
        sampling_rates=sampling_rates,
        feature_selections=feature_selections,
        decoding_model_choices=decoding_model_choices,
        random_seeds=random_seeds,
        filterings=filterings,
        cuda_id_list=[0],
    )

    # print time elapsed
    logging.info(f'Time elapsed: {time.time() - start_time:.2f}s')

    # 5269.65s for one model across seeds and sampling rates
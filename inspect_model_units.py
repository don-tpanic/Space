import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import time
import logging
import itertools
import multiprocessing

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import utils
import data
import models
import scores

"""
Visualize individual model units see 
if there are any patterns (e.g., place cells.)
"""

def _is_dead_unit(heatmap):
    """
    Given a unit's 2D heatmap, check if it is a dead unit
    by checking if all values in heatmap are zero/near-zero.
    """
    return np.allclose(heatmap, 0)


def _single_model_reps(config):
    """
    Produce model_reps either directly computing if the first time,
    or load from disk if already computed.

    return:
        model_reps: \in (n_locations, n_rotations, n_features)
    """
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{TF_NUM_INTRAOP_THREADS}"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    
    model_reps_fname = \
        f'results/'\
        f'{config["unity_env"]}/'\
        f'{config["movement_mode"]}/'\
        f'{config["model_name"]}/'\
        f'model_reps_{config["output_layer"]}.npy'

    if os.path.exists(model_reps_fname):
        logging.info(f'Loading model_reps from {model_reps_fname}')
        model_reps = np.load(model_reps_fname)
        logging.info(f'model_reps.shape: {model_reps.shape}')
        return model_reps

    else:
        # load model outputs
        if config['model_name'] == 'none':
            model = None
            preprocess_func = None
        else:
            model, preprocess_func = models.load_model(
                config['model_name'], config['output_layer'])

        preprocessed_data = data.load_preprocessed_data(
            config=config,
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

        # save to disk  
        # # TODO: slow to save but is it benefitcial as we do it only once?
        # logging.info(f'Saving model_reps to {model_reps_fname}...')
        # np.save(model_reps_fname, model_reps)
        # logging.info(f'[Saved] model_reps to {model_reps_fname}')
        return model_reps


def _single_env_viz_units_ranked_by_coef(
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
    Plot individual model units as heatmaps both 
        1. each rotation independently and
        2. summed over rotations.
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

        # Due to meeting 24-May-2023, we use absolute
        # values of coef for filtering.
        coef = np.abs(coef)
        for target_index in range(coef.shape[0]):
            # filter columns of `model_reps` 
            # based on each coef of each target
            # based on `n_units_filtering` and `filtering_order`
            for filtering in filterings:
                n_units_filtering = filtering['n_units_filtering']
                filtering_order = filtering['filtering_order']

                if filtering_order == 'top_n':
                    filtered_n_units_indices = np.argsort(
                        coef[target_index, :])[::-1][:n_units_filtering]

                elif filtering_order == 'mid_n':
                    filtered_n_units_indices = np.argsort(
                        coef[target_index, :])[::-1][
                            int(coef.shape[1]/2)-int(n_units_filtering/2):
                            int(coef.shape[1]/2)+int(n_units_filtering/2)]
                    
                elif filtering_order == 'random_n':
                    # randomly sample n_units_filtering units
                    # but excluding the top_n (also n_units_filtering)
                    np.random.seed(random_seed)
                    filtered_n_units_indices = np.random.choice(
                        np.argsort(
                            coef[target_index, :])[::-1][n_units_filtering:],
                            n_units_filtering,
                            replace=False)
                else:
                    raise NotImplementedError

                # fig, axes = plt.subplots(
                #     nrows=n_units_filtering, 
                #     ncols=model_reps.shape[1], 
                #     figsize=(600, 600)
                # )
                # for unit_rank, unit_index in enumerate(filtered_n_units_indices):
                #     for rotation in range(model_reps.shape[1]):
                #         if movement_mode == '2d':
                #             # reshape to (n_locations, n_rotations, n_features)
                #             heatmap = model_reps[:, rotation, unit_index].reshape(
                #                 (env_x_max*multiplier-env_x_min*multiplier+1, 
                #                 env_y_max*multiplier-env_y_min*multiplier+1)
                #             )

                #             # rotate heatmap to match Unity coordinate system
                #             # ref: tests/testReshape_forHeatMap.py
                #             heatmap = np.rot90(heatmap, k=1, axes=(0, 1))

                #             # plot heatmap
                #             axes[unit_rank, rotation].imshow(heatmap)
                #             axes[-1, rotation].set_xlabel('Unity x-axis')
                #             axes[unit_rank, 0].set_ylabel('Unity z-axis')
                #             axes[unit_rank, rotation].set_xticks([])
                #             axes[unit_rank, rotation].set_yticks([])

                # sup_title = f"{filtering_order},{targets[target_index]}, "\
                #             f"{config['unity_env']},{movement_mode},"\
                #             f"{config['model_name']},{feature_selection}"\
                #             f"({decoding_model_choice['hparams']}),"\
                #             f"sr{sampling_rate},seed{random_seed}"

                # figs_path = utils.load_figs_path(
                #     config=config,
                #     experiment=experiment,
                #     reference_experiment=reference_experiment,
                #     feature_selection=feature_selection,
                #     decoding_model_choice=decoding_model_choice,
                #     sampling_rate=sampling_rate,
                #     moving_trajectory=moving_trajectory,
                #     random_seed=random_seed,
                # )

                # # fix suptitle overlap.
                # # fig.tight_layout(rect=[0, 0.03, 1, 0.98])
                # plt.suptitle(sup_title)
                # plt.savefig(
                #     f'{figs_path}/units_heatmaps_{targets[target_index]}'\
                #     f'_{filtering_order}.png'
                # )
                # plt.close()
                # logging.info(
                #     f'[Saved] units heatmaps {targets[target_index]}'\
                #     f'{filtering_order} to {figs_path}'
                # )

                # plot summed over rotation heatmap and distribution of loc-wise
                # activation intensities.
                model_reps_summed = np.sum(
                    model_reps, axis=1, keepdims=True)

                # 1 for heatmap, 1 for distribution
                fig, axes = plt.subplots(
                    nrows=n_units_filtering, 
                    ncols=3,
                    figsize=(5, 600)
                )
                for unit_rank, unit_index in enumerate(filtered_n_units_indices):
                    for rotation in range(model_reps_summed.shape[1]):
                        if movement_mode == '2d':
                            # reshape to (n_locations, n_rotations, n_features)
                            heatmap = model_reps_summed[:, rotation, unit_index].reshape(
                                (env_x_max*multiplier-env_x_min*multiplier+1, 
                                env_y_max*multiplier-env_y_min*multiplier+1)
                            )

                            # rotate heatmap to match Unity coordinate system
                            # ref: tests/testReshape_forHeatMap.py
                            heatmap = np.rot90(heatmap, k=1, axes=(0, 1))

                            # compute fields info and write them on the heatmap
                            num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
                                mean_value_in_clusters, var_value_in_clusters, \
                                    bounds_heatmap = \
                                        _compute_single_heatmap_fields_info(
                                            heatmap=heatmap,
                                            pixel_min_threshold=10,
                                            pixel_max_threshold=int(heatmap.shape[0]*heatmap.shape[1]*0.5)
                                        )
                            
                            # plot heatmap on the left column.
                            axes[unit_rank, 0].imshow(heatmap)
                            axes[-1, 0].set_xlabel('Unity x-axis')
                            axes[-1, 0].set_ylabel('Unity z-axis')
                            axes[unit_rank, 0].set_xticks([])
                            axes[unit_rank, 0].set_yticks([])
                            axes[unit_rank, 0].set_title(
                                f'rank{unit_rank},u{unit_index}\n'\
                                f'coef{coef[target_index, unit_index]:.1f}'\
                                f'{num_clusters},{num_pixels_in_clusters},{max_value_in_clusters} '\
                            )

                            # plot heatmap contour on the middle column.
                            axes[unit_rank, 1].imshow(bounds_heatmap)

                            # plot distribution on the right column.
                            axes[unit_rank, 2].hist(
                                model_reps_summed[:, rotation, unit_index],
                                bins=10,
                            )
                            axes[-1, 2].set_xlabel('Activation intensity')

                sup_title = f"{filtering_order},{targets[target_index]},"\
                            f"{config['unity_env']},{movement_mode},"\
                            f"{config['model_name']},{feature_selection}"\
                            f"({decoding_model_choice['hparams']}),"\
                            f"sr{sampling_rate},seed{random_seed}"
                
                figs_path = utils.load_figs_path(
                    config=config,
                    experiment=experiment,
                    reference_experiment=reference_experiment,
                    feature_selection=feature_selection,
                    decoding_model_choice=decoding_model_choice,
                    sampling_rate=sampling_rate,
                    moving_trajectory=moving_trajectory,
                    random_seed=random_seed,
                )
                # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.tight_layout()
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


def _compute_single_heatmap_fields_info(
        heatmap, 
        pixel_min_threshold, 
        pixel_max_threshold
    ):
    """
    Given a 2D heatmap of a unit, compute:
        num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
            mean_value_in_clusters, var_value_in_clusters, heatmap_thresholded
    """
    scaler = MinMaxScaler()
    # normalize to [0, 1]
    heatmap_normalized = scaler.fit_transform(heatmap)  
    # convert to [0, 255]      
    heatmap_gray = (heatmap_normalized * 255).astype(np.uint8)
    # compute activity threshold as the mean of the heatmap
    activity_threshold = np.mean(heatmap_gray)

    _, heatmap_thresholded = cv2.threshold(
        heatmap_gray, activity_threshold, 
        255, cv2.THRESH_BINARY
    )

    # num_labels=4,
    # num_labels includes background
    # labels \in (17, 17)
    # stats \in (4, 5): [left, top, width, height, area] for each label
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(heatmap_thresholded)

    # Create a mask to filter clusters based on pixel thresholds
    # e.g. mask=[False, True, False, True] for each label (i.e. a cluster)
    mask = (stats[:, cv2.CC_STAT_AREA] >= pixel_min_threshold) & \
            (stats[:, cv2.CC_STAT_AREA] <= pixel_max_threshold)
    # set background to False regardless of pixel thresholds
    mask[0] = False
        
    # Filter the stats and labels based on the mask
    # filtered_stats.shape (2, 5)
    filtered_stats = stats[mask]

    # For labels with mask=True, keep the label, otherwise set to 0
    # this in fact will include 0, but we want 1, 3 only
    # so when using `filtered_labels` to extract max value in each cluster
    # we need to exclude 0
    filtered_labels = np.where(np.isin(labels, np.nonzero(mask)[0]), labels, 0)

    # Count the number of clusters that meet the criteria
    num_clusters = np.array([filtered_stats.shape[0]])

    # Get the number of pixels in each cluster
    num_pixels_in_clusters = filtered_stats[:, cv2.CC_STAT_AREA]

    # Get the max/mean/var value in heatmap based on each cluster
    max_value_in_clusters = []
    mean_value_in_clusters = []
    var_value_in_clusters = []
    for label in np.unique(filtered_labels):
        if label != 0:
            max_value_in_clusters.append(
                np.around(
                    np.max(heatmap[filtered_labels == label]), 1
                )
            )
            mean_value_in_clusters.append(
                np.around(
                    np.mean(heatmap[filtered_labels == label]), 1
                )
            )
            var_value_in_clusters.append(
                np.around(
                    np.var(heatmap[filtered_labels == label]), 1
                )
            )
            
    # Add 0 to `num_pixels_in_clusters` and `max_value_in_clusters`
    # in case `num_clusters` is 0. This is helpful when we want to
    # plot fields info against coef, as no matter if there is a cluster
    # for a unit, there is always a coef for that unit.
    if num_clusters[0] == 0:
        num_pixels_in_clusters = np.array([0])
        max_value_in_clusters = np.array([0])
        mean_value_in_clusters = np.array([0])
        var_value_in_clusters = np.array([0])
    else:
        max_value_in_clusters = np.array(max_value_in_clusters)
        mean_value_in_clusters = np.array(mean_value_in_clusters)
        var_value_in_clusters = np.array(var_value_in_clusters)

    colors = np.arange(100, dtype=int).tolist()
    for label in np.unique(filtered_labels):
        if label != 0:
            # create a mask for each label
            mask = np.where(filtered_labels == label, 255, 0).astype(np.uint8)
            # find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # draw contours
            cv2.drawContours(heatmap_thresholded, contours, -1, colors[label-1], 1)

    return num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
        mean_value_in_clusters, var_value_in_clusters, heatmap_thresholded


def _compute_single_heatmap_grid_scores(activation_map, smooth=False):
    # mask parameters
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())

    scorer = scores.GridScorer(
        len(activation_map),        # nbins
        [0, len(activation_map)-1], # coords_range
        masks_parameters            # parameters for the masks
    )

    score_60, score_90, max_60_mask, max_90_mask, sac = \
        scorer.get_scores(activation_map)
    
    return score_60, score_90, max_60_mask, max_90_mask, sac, scorer


def _compute_single_heatmap_border_scores(activation_map, db=3):
    """
    Banino et al. 2018 uses db=3.
    """
    num_bins = activation_map.shape[0]
    
    # Compute c (average activity for bins further than db bins from any wall)
    c = np.mean([
        activation_map[i, j]
        for i in range(db, num_bins - db)
        for j in range(db, num_bins - db)
    ])

    wall_scores = []

    # Compute the average activation for each wall
    for i in range(4):
        if i == 0:
            # Top wall
            activations = activation_map[:db, :]
        elif i == 1:
            # Right wall
            activations = activation_map[:, -db:]
        elif i == 2:
            # Bottom wall
            activations = activation_map[-db:, :]
        elif i == 3:
            # Left wall
            activations = activation_map[:, :db]

        bi = np.mean(activations)
        wall_scores.append((bi - c) / (bi + c))

    return np.max(wall_scores)


def _single_env_produce_fields_info_ranked_by_coef(
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
    Produce fields info for each unit and save to disk, which 
    will be used for plotting by `_single_env_viz_fields_info`.
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

        # Due to meeting 24-May-2023, we use absolute
        # values of coef for filtering.
        coef = np.abs(coef)
        for target_index in range(coef.shape[0]):
            # filter columns of `model_reps` 
            # based on each coef of each target
            # based on `n_units_filtering` and `filtering_order`
            for filtering in filterings:
                n_units_filtering = filtering['n_units_filtering']
                filtering_order = filtering['filtering_order']

                if filtering_order == 'top_n':
                    filtered_n_units_indices = np.argsort(
                        coef[target_index, :])[::-1][:n_units_filtering]

                elif filtering_order == 'mid_n':
                    filtered_n_units_indices = np.argsort(
                        coef[target_index, :])[::-1][
                            int(coef.shape[1]/2)-int(n_units_filtering/2):
                            int(coef.shape[1]/2)+int(n_units_filtering/2)]
                    
                elif filtering_order == 'random_n':
                    # randomly sample n_units_filtering units
                    # but excluding the top_n (also n_units_filtering)
                    np.random.seed(random_seed)
                    filtered_n_units_indices = np.random.choice(
                        np.argsort(
                            coef[target_index, :])[::-1][n_units_filtering:],
                            n_units_filtering,
                            replace=False)
                else:
                    raise NotImplementedError

                # fields info for each unit is computed on the summed heatmap 
                # across rotations.
                model_reps_summed = np.sum(model_reps, axis=1, keepdims=True)
                for unit_rank, unit_index in enumerate(filtered_n_units_indices):
                    for rotation in range(model_reps_summed.shape[1]):
                        if movement_mode == '2d':
                            # reshape to (n_locations, n_rotations, n_features)
                            heatmap = model_reps_summed[:, rotation, unit_index].reshape(
                                (env_x_max*multiplier-env_x_min*multiplier+1, 
                                env_y_max*multiplier-env_y_min*multiplier+1)
                            )

                            # rotate heatmap to match Unity coordinate system
                            # ref: tests/testReshape_forHeatMap.py
                            heatmap = np.rot90(heatmap, k=1, axes=(0, 1))
                            
                            # compute, collect and save fields info
                            unit_fields_info = []
                            num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
                                mean_value_in_clusters, var_value_in_clusters, \
                                    bounds_heatmap = \
                                        _compute_single_heatmap_fields_info(
                                            heatmap=heatmap,
                                            pixel_min_threshold=10,
                                            pixel_max_threshold=int(heatmap.shape[0]*heatmap.shape[1]*0.5)
                                        )
                            unit_fields_info.append(num_clusters)
                            unit_fields_info.append(num_pixels_in_clusters)
                            unit_fields_info.append(max_value_in_clusters)
                            unit_fields_info.append(mean_value_in_clusters)
                            unit_fields_info.append(var_value_in_clusters)
                            unit_fields_info.append(np.array([np.mean(heatmap)]))
                            unit_fields_info.append(np.array([np.var(heatmap)]))

                            # NOTE: we then save this unit's coef per target dimension
                            # as the last item in the list of fields info. 
                            # by doing this, we can easily access the coef of this saved
                            # ranked unit without having to load `coef.npy` which is 
                            # quite cumbersome.
                            # NOTE: in order to plot coef(x-axis) v fields info(y-axis), we need to 
                            # make sure coef is repeated the same number of times as the number of
                            # clusters.
                            if num_clusters[0] > 1:
                                unit_fields_info.append(
                                    np.array(
                                        coef[target_index, unit_index].repeat(num_clusters[0])
                                    )
                                )
                            else:
                                unit_fields_info.append(
                                    np.array(
                                        [coef[target_index, unit_index]])
                                )
                            
                            unit_fields_info = np.array(
                                unit_fields_info, dtype=object
                            )

                            # save each unit fields info to disk
                            results_path = utils.load_results_path(
                                config=config,
                                experiment='fields_info',
                                reference_experiment=reference_experiment,
                                feature_selection=feature_selection,
                                decoding_model_choice=decoding_model_choice,
                                sampling_rate=sampling_rate,
                                moving_trajectory=moving_trajectory,
                                random_seed=random_seed,
                            )
                            fpath = f'{results_path}/'\
                                    f'{filtering_order}'\
                                    f'_rank{unit_rank}'\
                                    f'_{targets[target_index]}.npy'
                            logging.info(f'Saving unit fields info to {fpath}')
                            np.save(fpath, unit_fields_info)


def _single_env_viz_fields_info_ranked_by_coef(
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
    Visualize fields info across units using fields info of units 
    produced by `_single_env_produce_fields_info`.
    
    1. So far we are tracking the following info related to each unit: 
        tracked_fields_info = [
            'num_clusters', 'num_pixels_in_clusters', 
            'max_value_in_clusters', 'mean_value_in_clusters', 'var_value_in_clusters', 
            'entire_map_mean', 'entire_map_var'
        ]

    2. The fields info for each unit is stored in a list of lists:
        fields_info = [[num_clusters], [num_pixels_in_clusters], [max_value_in_clusters], ...]

    3. As we created separate groups of units based on top_n based 
        on units' corresponding coef (abs due to Meeting 25-May-2023). 
        We can plot these units using their fields info as representations, 
        and see if there are patterns associated with
        whether these units are from the top_n or other groups (random_n, mid_n, etc.)

        We can look at a few things:
            1. num_clusters across top_n vs bottom_n
            2. num_pixels_in_clusters across top_n vs bottom_n
            3. max_value_in_clusters across top_n vs bottom_n
            4. ...
    
    4. Also perhaps across layers these fields info differ and can help 
        us understand the difference in decoding performance.
    """
    targets = ['x', 'y', 'rot']
    tracked_fields_info = [
        'num_clusters', 'num_pixels_in_clusters', 
        'max_value_in_clusters', 'mean_value_in_clusters', 'var_value_in_clusters', 
        'entire_map_mean', 'entire_map_var']
    n_units_filtering = filterings[0]['n_units_filtering']
    filtering_types = [f['filtering_order'] for f in filterings]
    
    config = utils.load_config(config_version)
    results_path = utils.load_results_path(
        config=config,
        experiment=experiment,
        reference_experiment=reference_experiment,
        feature_selection=feature_selection,
        decoding_model_choice=decoding_model_choice,
        sampling_rate=sampling_rate,
        moving_trajectory=moving_trajectory,
        random_seed=random_seed,
    )

    if results_path is None:
        logging.info(
            f'Mismatch between feature '\
            f'selection and decoding model, skip.'
        )
        return
    
    for target in targets:
        fig, axes = plt.subplots(
            nrows=len(tracked_fields_info), 
            ncols=3, 
            figsize=(14, 14)
        )
        for info_index, info in enumerate(tracked_fields_info):
            top_n_stats = []
            top_n_coef = []
            random_n_stats = []
            random_n_coef = []
            mid_n_stats = []
            mid_n_coef = []
            for filtering in filtering_types:
                for unit_rank in range(n_units_filtering):
                    fname = f'{results_path}/{filtering}_rank{unit_rank}_{target}.npy'
                    try:
                        fields_info = np.load(fname, allow_pickle=True)
                    except FileNotFoundError:
                        logging.info(
                            f'File {fname} not found, must `_single_env_viz_units`'\
                            f'to save the units fields info first.'
                        )
                        exit
                    
                    if info == 'num_clusters':
                        stats = fields_info[0]
                    elif info == 'num_pixels_in_clusters':
                        stats = fields_info[1]
                    elif info == 'max_value_in_clusters':
                        stats = fields_info[2]
                    elif info == 'mean_value_in_clusters':
                        stats = fields_info[3]
                    elif info == 'var_value_in_clusters':
                        stats = fields_info[4]
                    elif info == 'entire_map_mean':
                        stats = fields_info[5]
                    elif info == 'entire_map_var':
                        stats = fields_info[6]

                    if filtering == 'top_n':
                        top_n_stats.extend(stats)
                        if info in ['num_clusters', 
                                    'entire_map_mean', 
                                    'entire_map_var']:
                            # HACK: due to coef is repeated the same number of times
                            # as the number of clusters during saving, but when 
                            # info == 'num_clusters', there is only one value for 
                            # one coef, we need to extract the first coef from the list
                            # of coef (first coef is the same as the rest though)
                            top_n_coef.append(fields_info[-1][0])
                        else:
                            top_n_coef.extend(fields_info[-1])
                    
                    elif filtering == 'mid_n':
                        mid_n_stats.extend(stats)
                        if info in ['num_clusters', 
                                    'entire_map_mean', 
                                    'entire_map_var']:
                            mid_n_coef.append(fields_info[-1][0])
                        else:
                            mid_n_coef.extend(fields_info[-1])
                    
                    elif filtering == 'random_n':
                        random_n_stats.extend(stats)
                        if info in ['num_clusters', 
                                    'entire_map_mean', 
                                    'entire_map_var']:
                            random_n_coef.append(fields_info[-1][0])
                        else:
                            random_n_coef.extend(fields_info[-1])
                    
            # plot for each info, how units/fields differ
            # set x-axis correctly as they differ for different info
            # e.g. num_clusters are wrt units, whereas max_value_in_clusters
            # are wrt clusters (longer axis due to each unit may have multiple clusters)
            if info in ['num_clusters', 'entire_map_mean', 'entire_map_var']:
                axes[info_index, 0].set_xlabel('units')
            elif info in ['num_pixels_in_clusters', 'max_value_in_clusters', 
                          'mean_value_in_clusters', 'var_value_in_clusters']:
                axes[info_index, 0].set_xlabel('fields')

            axes[info_index, 0].set_title(info)
            axes[info_index, 0].plot(
                np.arange(len(top_n_stats)), top_n_stats, label='top_n', alpha=0.5
            )
            axes[info_index, 0].plot(
                np.arange(len(mid_n_stats)), mid_n_stats, label='mid_n', alpha=0.5,
            )
            axes[info_index, 0].plot(
                np.arange(len(random_n_stats)), random_n_stats, label='random_n', alpha=0.5,
                c='gray'
            )
            
            # kdeplot for each info, how units (distribution) differ
            axes[info_index, 1].set_title(info)
            sns.kdeplot(
                top_n_stats, label='top_n', ax=axes[info_index, 1], alpha=0.5
            )
            sns.kdeplot(
                mid_n_stats, label='mid_n', ax=axes[info_index, 1], alpha=0.5,
            )
            sns.kdeplot(
                random_n_stats, label='random_n', ax=axes[info_index, 1], alpha=0.5,
                color='gray'
            )

            # scatterplot for each info, how unit coef and info correlate
            axes[info_index, 2].set_title(info)
            axes[info_index, 2].scatter(
                top_n_coef, top_n_stats, label='top_n', alpha=0.1
            )
            axes[info_index, 2].scatter(
                mid_n_coef, mid_n_stats, label='mid_n', alpha=0.1,
            )
            axes[info_index, 2].scatter(
                random_n_coef, random_n_stats, label='random_n', alpha=0.3,
                c='gray', marker='x'
            )
            axes[info_index, 2].set_xlabel('coef')


        sup_title = f"{target},"\
                    f"{config['unity_env']},"\
                    f"{config['model_name']},{feature_selection}"\
                    f"({decoding_model_choice['hparams']}),"\
                    f"sr{sampling_rate},seed{random_seed}"
                
        figs_path = utils.load_figs_path(
            config=config,
            experiment=experiment,
            reference_experiment=reference_experiment,
            feature_selection=feature_selection,
            decoding_model_choice=decoding_model_choice,
            sampling_rate=sampling_rate,
            moving_trajectory=moving_trajectory,
            random_seed=random_seed,
        )

        plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(sup_title)
        plt.savefig(
            f'{figs_path}/units_fields_info'\
            f'_{target}_groupedByFilteringN.png')
        plt.close()
        logging.info(
            f'[Saved] {figs_path}/units_fields_info'\
            f'_{target}_groupedByFilteringN.png')


def _single_env_produce_unit_chart(
        config_version, 
        experiment,
        moving_trajectory,
        reference_experiment=None,
        feature_selection=None, 
        decoding_model_choice=None,
        sampling_rate=None,
        random_seed=None,
        filterings=None,  
        # charting all units, use Nones to maintain API consistency
    ):
    """
    Produce unit chart for each unit and save to disk, which 
    will be used for plotting by `_single_env_viz_unit_chart`.

    Unit chart is intended to capture characteristics of each 
    unit (no filtering; ALL units). Currently, the chart includes:
        0. if dead (if true, continue to next unit)
        1. fields info - [
                            num_clusters, 
                            num_pixels_in_clusters, 
                            max_value_in_clusters, 
                            mean_value_in_clusters, 
                            var_value_in_clusters,
                            entire_map_mean,
                            entire_map_var,
                        ]
        2. gridness
        3. borderness
        4. directioness TODO: but how to separate from the rest as it is rotation specific?
    """
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{TF_NUM_INTRAOP_THREADS}"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    config = utils.load_config(config_version)
    movement_mode=config['movement_mode']
    env_x_min=config['env_x_min']
    env_x_max=config['env_x_max']
    env_y_min=config['env_y_min']
    env_y_max=config['env_y_max']
    multiplier=config['multiplier']
    
    # load model outputs
    model_reps = _single_model_reps(config)

    # charted info:
    charted_info = [
                    'dead',
                    'num_clusters', 
                    'num_pixels_in_clusters', 
                    'max_value_in_clusters', 
                    'mean_value_in_clusters', 
                    'var_value_in_clusters',
                    'entire_map_mean',
                    'entire_map_var',
                    'gridness',
                    'borderness',
                    # 'coef', # TODO: coef not charted but can be combined later.
                ]
    
    # initialize unit chart collector
    # shape \in (total_n_units, len(charted_info))
    # init from zero so as we first check if a unit is dead
    # if dead, we continue to next unit so the rest of the info re this unit 
    # will be kept as zero
    unit_chart_info = np.zeros(
        (model_reps.shape[2], len(charted_info)), dtype=object
    )
    model_reps_summed = np.sum(
        model_reps, axis=1, keepdims=True
    )

    for unit_index in range(model_reps_summed.shape[2]):
        for rotation in range(model_reps_summed.shape[1]):
            logging.info(f'[Charting] unit_index: {unit_index}')
            if movement_mode == '2d':
                # reshape to (n_locations, n_rotations, n_features)
                heatmap = model_reps_summed[:, rotation, unit_index].reshape(
                    (env_x_max*multiplier-env_x_min*multiplier+1, 
                    env_y_max*multiplier-env_y_min*multiplier+1) )
                # rotate heatmap to match Unity coordinate system
                # ref: tests/testReshape_forHeatMap.py
                heatmap = np.rot90(heatmap, k=1, axes=(0, 1))

                ###### Go thru each required info, maybe modularise later.
                if _is_dead_unit(heatmap):
                    unit_chart_info[unit_index, 0] = np.array([0])
                    continue
                else:
                    unit_chart_info[unit_index, 0] = np.array([1])
                    # compute, collect and save unit chart info
                    # 1. fields info
                    num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
                        mean_value_in_clusters, var_value_in_clusters, \
                            bounds_heatmap = \
                                _compute_single_heatmap_fields_info(
                                    heatmap=heatmap,
                                    pixel_min_threshold=10,
                                    pixel_max_threshold=\
                                    int(heatmap.shape[0]*heatmap.shape[1]*0.5)
                                )
                    unit_chart_info[unit_index, 1] = num_clusters
                    unit_chart_info[unit_index, 2] = num_pixels_in_clusters
                    unit_chart_info[unit_index, 3] = max_value_in_clusters
                    unit_chart_info[unit_index, 4] = mean_value_in_clusters
                    unit_chart_info[unit_index, 5] = var_value_in_clusters
                    unit_chart_info[unit_index, 6] = np.array([np.mean(heatmap)])
                    unit_chart_info[unit_index, 7] = np.array([np.var(heatmap)])

                    # 2. gridness
                    score_60_, _, _, _, sac, scorer = \
                                _compute_single_heatmap_grid_scores(heatmap)
                    unit_chart_info[unit_index, 8] = score_60_

                    # 3. borderness
                    border_score = _compute_single_heatmap_border_scores(heatmap)
                    unit_chart_info[unit_index, 9] = border_score
        
    results_path = utils.load_results_path(
        config=config,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    fpath = f'{results_path}/unit_chart.npy'
    np.save(fpath, unit_chart_info)
    logging.info(f'[Saved] {fpath}')


def _single_env_viz_gridness_ranked_by_unit_chart(
        config_version, 
        experiment,
        moving_trajectory,
        reference_experiment=None,
        feature_selection=None, 
        decoding_model_choice=None,
        sampling_rate=None,
        random_seed=None,
        filterings=\
            [{'filtering_order': 'top_n', 'n_units_filtering': 400},
             {'filtering_order': 'mid_n', 'n_units_filtering': 400},
             {'filtering_order': 'random_n', 'n_units_filtering': 400},
            ]
    ):
    """
    Based on unit chart info produced by `_single_env_produce_unit_chart`,
    We can load the chart and sort by unit gridness and visualize the
    top_n, mid_n,  gridness units in terms of ratemaps and autocorrelagrams.

    Notice the postfix `sorted_from_unit_chart` is to distinguish from 
    `*viz_fields_info*` and `*viz_units*` which are based on coef ranking 
    which are based on specific combination of feature selection, sampling rate,
    etc. Whereas here the visualization is general to all the settings and are
    not coef-based but gridness-based (i.e. filtering is based on gridness).

    We could replace coef-based viz for fields_info and switch to `fields`-based
    ranking but that would require a specific selection criterion such as `num_clusters`
    which we might get to at a later stage.
    """
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{TF_NUM_INTRAOP_THREADS}"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    config = utils.load_config(config_version)
    movement_mode=config['movement_mode']
    env_x_min=config['env_x_min']
    env_x_max=config['env_x_max']
    env_y_min=config['env_y_min']
    env_y_max=config['env_y_max']
    multiplier=config['multiplier']
    
    # load model outputs
    model_reps = _single_model_reps(config)
    model_reps_summed = np.sum(model_reps, axis=1, keepdims=True)
    
    # load unit chart info
    results_path = utils.load_results_path(
        config=config,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    unit_chart_info = np.load(f'{results_path}/unit_chart.npy', allow_pickle=True)
    gridnesses = unit_chart_info[:, 8]
    
    # visualize top_n, mid_n, random_n units' gridness
    for filtering in filterings:
        n_units_filtering = filtering['n_units_filtering']
        filtering_order = filtering['filtering_order']

        logging.info(f'gridnesses.shape: {gridnesses.shape}')
        logging.info(f'filtering_order: {filtering_order}')

        if filtering_order == 'top_n':
            filtered_n_units_indices = np.argsort(gridnesses)[::-1][:n_units_filtering]

        elif filtering_order == 'mid_n':
            filtered_n_units_indices = np.argsort(gridnesses)[::-1][
                    int(gridnesses.shape[0]/2)-int(n_units_filtering/2):
                    int(gridnesses.shape[0]/2)+int(n_units_filtering/2)]
            
        elif filtering_order == 'random_n':
            # randomly sample n_units_filtering units
            # but excluding the top_n (also n_units_filtering)
            np.random.seed(random_seed)
            filtered_n_units_indices = np.random.choice(
                np.argsort(gridnesses)[::-1][n_units_filtering:],
                    n_units_filtering,
                    replace=False)
        else:
            raise NotImplementedError

        # plotter
        fig, axes = plt.subplots(
            nrows=n_units_filtering, ncols=2, figsize=(5, 600)
        )

        for row_index, unit_index in enumerate(filtered_n_units_indices):
            for rotation in range(model_reps_summed.shape[1]):
                if movement_mode == '2d':
                    # reshape to (n_locations, n_rotations, n_features)
                    heatmap = model_reps_summed[:, rotation, unit_index].reshape(
                        (env_x_max*multiplier-env_x_min*multiplier+1, 
                        env_y_max*multiplier-env_y_min*multiplier+1) )
                    # rotate heatmap to match Unity coordinate system
                    # ref: tests/testReshape_forHeatMap.py
                    heatmap = np.rot90(heatmap, k=1, axes=(0, 1))

                    # plot heatmap for the selected units
                    axes[row_index, 0].imshow(heatmap)
                    axes[row_index, 0].set_title(f'unit:{unit_index}')
                    axes[row_index, 0].set_xticks([])
                    axes[row_index, 0].set_yticks([])

                    # plot autocorrelagram for the selected units
                    _, _, _, _, sac, scorer = \
                                _compute_single_heatmap_grid_scores(heatmap)
                    
                    useful_sac = sac * scorer._plotting_sac_mask
                    axes[row_index, 1].imshow(useful_sac)
                    axes[row_index, 1].set_title(f'gridness:{gridnesses[unit_index]:.2f}')
                    axes[row_index, 1].set_xticks([])
                    axes[row_index, 1].set_yticks([])

        figs_path = utils.load_figs_path(
            config=config,
            experiment=experiment,
            moving_trajectory=moving_trajectory,
        )
        plt.savefig(
            f'{figs_path}/gridness_sorted_from_unit_chart_{filtering_order}.png'
        )
        plt.close()
        logging.info(
            f'[Saved] {figs_path}/gridness_sorted_from_unit_chart_{filtering_order}.png'
        )


def _single_env_viz_borderness_ranked_by_unit_chart(
        config_version, 
        experiment,
        moving_trajectory,
        reference_experiment=None,
        feature_selection=None, 
        decoding_model_choice=None,
        sampling_rate=None,
        random_seed=None,
        filterings=\
            [{'filtering_order': 'top_n', 'n_units_filtering': 400},
             {'filtering_order': 'mid_n', 'n_units_filtering': 400},
             {'filtering_order': 'random_n', 'n_units_filtering': 400},
            ]
    ):
    """
    """
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{TF_NUM_INTRAOP_THREADS}"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    config = utils.load_config(config_version)
    movement_mode=config['movement_mode']
    env_x_min=config['env_x_min']
    env_x_max=config['env_x_max']
    env_y_min=config['env_y_min']
    env_y_max=config['env_y_max']
    multiplier=config['multiplier']
    
    # load model outputs
    model_reps = _single_model_reps(config)
    model_reps_summed = np.sum(model_reps, axis=1, keepdims=True)
    
    # load unit chart info
    results_path = utils.load_results_path(
        config=config,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    unit_chart_info = np.load(f'{results_path}/unit_chart.npy', allow_pickle=True)
    borderness = unit_chart_info[:, 9]
    
    # visualize top_n, mid_n, random_n units' borderness
    for filtering in filterings:
        n_units_filtering = filtering['n_units_filtering']
        filtering_order = filtering['filtering_order']

        logging.info(f'borderness.shape: {borderness.shape}')
        logging.info(f'filtering_order: {filtering_order}')

        if filtering_order == 'top_n':
            filtered_n_units_indices = np.argsort(borderness)[::-1][:n_units_filtering]

        elif filtering_order == 'mid_n':
            filtered_n_units_indices = np.argsort(borderness)[::-1][
                    int(borderness.shape[0]/2)-int(n_units_filtering/2):
                    int(borderness.shape[0]/2)+int(n_units_filtering/2)]
            
        elif filtering_order == 'random_n':
            # randomly sample n_units_filtering units
            # but excluding the top_n (also n_units_filtering)
            np.random.seed(random_seed)
            filtered_n_units_indices = np.random.choice(
                np.argsort(borderness)[::-1][n_units_filtering:],
                    n_units_filtering,
                    replace=False)
        else:
            raise NotImplementedError

        # plotter
        fig, axes = plt.subplots(
            nrows=n_units_filtering, ncols=1, figsize=(5, 600)
        )

        for row_index, unit_index in enumerate(filtered_n_units_indices):
            for rotation in range(model_reps_summed.shape[1]):
                if movement_mode == '2d':
                    # reshape to (n_locations, n_rotations, n_features)
                    heatmap = model_reps_summed[:, rotation, unit_index].reshape(
                        (env_x_max*multiplier-env_x_min*multiplier+1, 
                        env_y_max*multiplier-env_y_min*multiplier+1) )
                    # rotate heatmap to match Unity coordinate system
                    # ref: tests/testReshape_forHeatMap.py
                    heatmap = np.rot90(heatmap, k=1, axes=(0, 1))

                    # plot heatmap for the selected units
                    axes[row_index].imshow(heatmap)
                    axes[row_index].set_xticks([])
                    axes[row_index].set_yticks([])
                    axes[row_index].set_title(
                        f'unit:{unit_index},'\
                        f'borderness:{borderness[unit_index]:.2f}'
                    )

        figs_path = utils.load_figs_path(
            config=config,
            experiment=experiment,
            moving_trajectory=moving_trajectory,
        )
        plt.savefig(
            f'{figs_path}/borderness_sorted_from_unit_chart_{filtering_order}.png'
        )
        plt.close()
        logging.info(
            f'[Saved] {figs_path}/borderness_sorted_from_unit_chart_{filtering_order}.png'
        )


def _single_env_viz_unit_chart(
        config_version, 
        experiment,
        moving_trajectory,
        reference_experiment=None,
        feature_selection=None, 
        decoding_model_choice=None,
        sampling_rate=None,
        random_seed=None,
        filterings=None,
    ):
    """
    Visualize unit chart info produced by `_single_env_produce_unit_chart`.

    For now, we chart:
        1. % units dead
        2. % units (place-cells) with 1, 2, .., max num_clusters
        3. % units (border-cells) with border firings..  # TODO: wait for border scoring.
    """
    # charted info
    charted_info = [
                    'dead',
                    'num_clusters', 
                    'num_pixels_in_clusters', 
                    'max_value_in_clusters', 
                    'mean_value_in_clusters', 
                    'var_value_in_clusters',
                    'entire_map_mean',
                    'entire_map_var',
                    'gridness',
                ]
    
    config = utils.load_config(config_version)
    
    # load unit chart info
    results_path = utils.load_results_path(
        config=config,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    unit_chart_info = np.load(
        f'{results_path}/unit_chart.npy', allow_pickle=True)
    logging.info(f'unit_chart_info.shape: {unit_chart_info.shape}')

    # Given `unit_chart_info`, for now we test `% dead`, `% num_clusters`
    #   For `% dead`, we iterate through unit_chart_info and count 
    # how many units are dead along the first column (i.e. unit_chart_info[i, 0] == 0)
    # we return the percentage of dead units.
    #   For `% num_clusters`, we iterate through unit_chart_info and count
    # how many units have 1, 2, .., max num_clusters.
    # and for each unique `num_clusters`, we return the percentage of corresponding units.
    #   For `% cluster size`, we iterate through unit_chart_info and count
    # the sizes of fields of qualified units.
    #   For `% peak cluster activation`, we iterate through unit_chart_info and count
    # the peak activation of fields of qualified units.
    dead_units = 0
    max_num_clusters = np.max(unit_chart_info[:, 1])
    num_clusters = np.zeros(max_num_clusters+1)
    cluster_sizes = []
    cluster_peaks = []

    for unit_index in range(unit_chart_info.shape[0]):
        if unit_chart_info[unit_index, 0] == 0:
            dead_units += 1
        else:
            num_clusters[int(unit_chart_info[unit_index, 1])] += 1
            cluster_sizes.extend(unit_chart_info[unit_index, 2])
            cluster_peaks.extend(unit_chart_info[unit_index, 3])

    
    # plot
    fig, axes = plt.subplots(
        nrows=4, ncols=1, figsize=(5, 10)
    )

    # 0-each bar is % of dead/active units
    # left bar is dead, right bar is active,
    # plot in gray for dead units, plot in blue for active units
    axes[0].bar(
        np.arange(2),
        [dead_units/unit_chart_info.shape[0],
            (unit_chart_info.shape[0]-dead_units)/unit_chart_info.shape[0]],
        color=['gray', 'blue']
    )
    axes[0].set_xticks(np.arange(2))
    axes[0].set_xticklabels(['dead', 'active'])
    axes[0].set_ylabel('% units')
    axes[0].set_title(f'% units dead/active')
    axes[0].set_ylim([-.05, 1.05])

    # 1-each bar is % of a num_clusters
    axes[1].bar(
        np.arange(max_num_clusters+1),
        num_clusters/unit_chart_info.shape[0]
    )
    axes[1].set_xlabel('num_clusters')
    axes[1].set_ylabel('% units')
    axes[1].set_title(f'% units with 1, 2, .., {max_num_clusters[0]} clusters')
    axes[1].set_ylim([-.05, 1.05])

    # 2-each bar is % of a cluster size (bined)
    axes[2].hist(
        cluster_sizes, bins=20, density=True
    )
    axes[2].set_xlabel('cluster size')
    axes[2].set_ylabel('density')
    axes[2].set_title(f'cluster size distribution')

    # 3-each bar is % of a cluster peak (bined)
    axes[3].hist(
        cluster_peaks, bins=20, density=True
    )
    axes[3].set_xlabel('cluster peak')
    axes[3].set_ylabel('density')
    axes[3].set_title(f'cluster peak distribution')
    
    figs_path = utils.load_figs_path(
        config=config,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    plt.tight_layout()
    plt.savefig(
        f'{figs_path}/unit_chart.png'
    )


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
    args_list = []
    for model_name in model_names:
        envs_dict = load_envs_dict(model_name, envs)
        config_versions=list(envs_dict.keys())
        # args_list = []
        for config_version in config_versions:
            for moving_trajectory in moving_trajectories:
                if experiment == 'unit_chart':
                    single_entry = {}
                    single_entry['config_version'] = config_version
                    single_entry['moving_trajectory'] = moving_trajectory
                    single_entry['experiment'] = experiment
                    args_list.append(single_entry)
                else:
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
    CPU_NUM_PROCESSES = 4     
    experiment = 'unit_chart'
    reference_experiment = None
    envs = ['env28_r24']
    movement_modes = ['2d']
    sampling_rates = [0.3]
    random_seeds = [42]
    model_names = ['vgg16']
    moving_trajectories = ['uniform']
    decoding_model_choices = []
    feature_selections = []
    filterings = []
    # ======================================== #
    
    multi_envs_inspect_units_GPU(
    # multi_envs_inspect_units_CPU(
        # target_func=_single_env_viz_units_ranked_by_coef,             # set experiment='viz'
        # target_func=_single_env_produce_fields_info_ranked_by_coef,   # set experiment='fields_info'
        # target_func=_single_env_viz_fields_info_ranked_by_coef,       # set experiment='fields_info'
        # target_func=_single_env_produce_unit_chart,                     # set experiment='unit_chart'
        # target_func=_single_env_viz_gridness_ranked_by_unit_chart,      # set experiment='unit_chart'
        target_func=_single_env_viz_borderness_ranked_by_unit_chart,    # set experiment='unit_chart'
        # target_func=_single_env_viz_unit_chart,                          # set experiment='unit_chart'
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
        cuda_id_list=[0, 1, 2, 3, 4, 5, 6, 7],
    )

    # print time elapsed
    logging.info(f'Time elapsed: {time.time() - start_time:.2f}s')
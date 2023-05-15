import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import time
import logging
import itertools
import multiprocessing
from sklearn.preprocessing import MinMaxScaler

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import utils
import data
import models

"""
Given model_reps, map back to 2D space, summed over
rotations, each 2D heatmap belongs to a model unit. 
For each unit, we apply metric-based place-cell approach
to find clusters and compute peak and number of fields.
"""

def _find_clusters(heatmap, pixel_min_threshold, pixel_max_threshold):

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
    num_clusters = filtered_stats.shape[0]

    # Get the number of pixels in each cluster
    num_pixels_in_clusters = filtered_stats[:, cv2.CC_STAT_AREA]

    # Get the max value in heatmap based on each cluster
    max_value_in_clusters = []
    for label in np.unique(filtered_labels):
        if label != 0:
            max_value_in_clusters.append(
                np.around(
                    np.max(heatmap[filtered_labels == label]), 1
                )
            )

    # TODO: do we actually need this?
    # # Draw contours of each cluster over the heatmap
    # # ref: https://stackoverflow.com/questions/8830619/difference-between-cv-retr-list-cv-retr-tree-cv-retr-external
    # for label in np.unique(filtered_labels):
    #     if label != 0:
    #         # find the contour of each cluster
    #         # contour \in (n_points, 1, 2)
    #         contour = cv2.findContours(
    #             (filtered_labels == label).astype(np.uint8), 
    #             cv2.RETR_CCOMP, 
    #             cv2.CHAIN_APPROX_SIMPLE
    #         )[0][0]
    #         # draw the contour on the heatmap
    #         cv2.drawContours(heatmap, [contour], -1, (255), 1)

    return num_clusters, num_pixels_in_clusters, max_value_in_clusters, filtered_labels


def _single_env_metric_units(
        config_version, 
        experiment,
        feature_selection, 
        decoding_model_choice,
        sampling_rate,
        moving_trajectory,
        random_seed,
        filterings,    
    ):
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{TF_NUM_INTRAOP_THREADS}"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"

    config = utils.load_config(config_version)
    n_rotations=config['n_rotations']
    movement_mode=config['movement_mode']
    env_x_min=config['env_x_min']
    env_x_max=config['env_x_max']
    env_y_min=config['env_y_min']
    env_y_max=config['env_y_max']
    multiplier=config['multiplier']
    results_path = utils.load_results_path(
        config=config,
        experiment='loc_n_rot',  # Dirty but coef is saved in loc_n_rot
        feature_selection=feature_selection,
        decoding_model_choice=decoding_model_choice,
        sampling_rate=sampling_rate,
        moving_trajectory=moving_trajectory,
        random_seed=random_seed,
    )
    logging.info(f'Loading results (for coef) from {results_path}')
    if results_path is None:
        logging.info(f'Mismatch between feature selection and decoding model, skip.')
        return

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
            f"{movement_mode}", 
        movement_mode=movement_mode,
        env_x_min=env_x_min,
        env_x_max=env_x_max,
        env_y_min=env_y_min,
        env_y_max=env_y_max,
        multiplier=multiplier,
        n_rotations=n_rotations,
        preprocess_func=preprocess_func,
    )    
    
    # (n_locations*n_rotations, n_features)
    model_reps = data.load_full_dataset_model_reps(
        config, model, preprocessed_data
    )

    # reshape to (n_locations, n_rotations, n_features)
    model_reps = model_reps.reshape(
        (model_reps.shape[0] // n_rotations,  # n_locations
        n_rotations,                          # n_rotations
        model_reps.shape[1])                            # all units
    )

    # prepare coordinates for plotting heatmap
    if movement_mode == '2d':
        x_axis_coords = []
        y_axis_coords = []
        # same idea as generating the frames in Unity
        # so we get decimal coords in between the grid points
        for i in range(env_x_min*multiplier, env_x_max*multiplier+1):
            for j in range(env_y_min*multiplier, env_y_max*multiplier+1):
                x_axis_coords.append(i/multiplier)
                y_axis_coords.append(j/multiplier)

    # ================================
    # TODO: to be integrated...
    # (n_locations, n_rotations, n_features)
    filtered_n_units_indices = np.array([1, 2, 3, 4, 5, 6])
    model_reps_filtered = model_reps[:, :, filtered_n_units_indices]

    # summed over rotations
    model_reps_filtered = np.sum(
        model_reps_filtered, axis=1, keepdims=True)

    # 1 for heatmap, 1 heatmap with contour, 1 for distribution
    fig, axes = plt.subplots(
        nrows=model_reps_filtered.shape[2], 
        ncols=3,
        figsize=(10, 25))
    
    for unit_index in range(model_reps_filtered.shape[2]):
        for rotation in range(model_reps_filtered.shape[1]):
            if movement_mode == '2d':
                # reshape to (n_locations, n_rotations, n_features)
                heatmap = model_reps_filtered[:, rotation, unit_index].reshape(
                    (env_x_max*multiplier-env_x_min*multiplier+1, 
                    env_y_max*multiplier-env_y_min*multiplier+1)
                )
                # plot the original heatmap from model_reps
                axes[unit_index, 0].imshow(heatmap)

                # count number of clusters and size of each cluster
                num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
                    heatmap_with_contours =  _find_clusters(
                        heatmap,
                        pixel_min_threshold=10,
                        pixel_max_threshold=\
                            int(heatmap.shape[0]*heatmap.shape[1]*0.5),
                )

                # plot on the mid column.
                axes[unit_index, 1].imshow(heatmap_with_contours)
                
                # write number of clusters and size of each cluster on each subplot
                axes[unit_index, 1].set_title(
                    f'unit {unit_index},\n'\
                    f'num_clusters: {num_clusters},\n' \
                    f'num_pixels_in_clusters: {num_pixels_in_clusters},\n'\
                    f'max_value_in_clusters: {max_value_in_clusters}'
                )

    plt.savefig('test.png')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    start_time = time.time()
    logging_level = 'info'
    if logging_level == 'info':
        logging.basicConfig(level=logging.INFO)
    elif logging_level == 'debug':
        logging.basicConfig(level=logging.DEBUG)

    # ======================================== #
    TF_NUM_INTRAOP_THREADS = 10
    CPU_NUM_PROCESSES = 30         
    experiment = 'viz'
    envs = ['env28_r24']
    movement_modes = ['2d']
    sampling_rates = [0.5]
    random_seeds = [42]
    model_names = ['vgg16']
    moving_trajectories = ['uniform']
    decoding_model_choices = [
        {'name': 'ridge_regression', 'hparams': 1.0},
    ]
    feature_selections = ['l2']
    filterings = [
        {'filtering_order': 'top_n', 'n_units_filtering': 20},
    ]
    # ======================================== #
    
    _single_env_metric_units(
        config_version='env28_r24_2d_vgg16_fc2', 
        experiment='viz',
        feature_selection='l2', 
        decoding_model_choice={'name': 'ridge_regression', 'hparams': 1.0},
        sampling_rate=0.5,
        moving_trajectory='uniform',
        random_seed=999,
        filterings=filterings, 
    )

    # print time elapsed
    logging.info(f'Time elapsed: {time.time() - start_time:.2f}s')
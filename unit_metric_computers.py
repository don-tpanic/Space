import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import logging
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import scores


def _is_dead_unit(heatmap):
    """
    Given a unit's 2D heatmap, check if it is a dead unit.
    """
    # return np.allclose(heatmap, 0)

    # unit is dead if less than 1% of the heatmap is active
    return np.sum(heatmap > 0) < 0.01 * heatmap.shape[0] * heatmap.shape[1]


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
    # scaler = MinMaxScaler()
    # # normalize to [0, 1]
    # heatmap_normalized = scaler.fit_transform(heatmap)
    heatmap_normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    
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


def _compute_single_heatmap_directional_scores(activation_maps):
    """
    Args:
        `activation_maps` correspond to the un-summed activation maps
        for a single unit across rotations \in (n_locations, n_rotations)

    - num of angular bins in Banino here becomes `n_rotations`.
    - based on Banino eq, we need to convert each n_rotations to 
        `alpha_i` which is angle.
    - the intensity `beta_i` of an angle is the average activation 
        across all locations for that angle.
    """
    # model_reps \in (n_locations, n_rotations, n_features)
    # activation_maps \in (n_locations, n_rotations)
    num_bins = activation_maps.shape[1]
    alphas = np.linspace(0, 2*np.pi, num=num_bins, endpoint=False)
    betas = np.mean(activation_maps, axis=0)

    # given a rotation, we can compute alpha_i and beta_i
    # which are used to compute r_i in the eq.
    # we collect r_i for each rotation and compute the mean
    # vector, whose length is used as the directional score.
    polar_plot_coords = [] # (n_rotations, 2)
    per_rotation_vector_length = []
    for alpha_i, beta_i in zip(alphas, betas):
        polar_plot_coords.append(
            [beta_i*np.cos(alpha_i), beta_i*np.sin(alpha_i)]
        )
        per_rotation_vector_length.append(
            np.linalg.norm([beta_i*np.cos(alpha_i), beta_i*np.sin(alpha_i)])
        )
    
    # to compute mean vector length,
    # first we compute the sum of r_i normed by sum of beta_i
    r_normed_by_beta = np.sum(
        np.array(polar_plot_coords), axis=0) / np.sum(betas)

    # then we compute the length of the mean vector
    mean_vector_length = np.linalg.norm(r_normed_by_beta)
    logging.info(f'[Check] mean_vector_length: {mean_vector_length}')
    return mean_vector_length, per_rotation_vector_length


def _unit_chart_type_classification(unit_chart_info):
    """
    Given a unit_chart_info, classify the units into different types,
    and return the indices of units by type or combo of types.
    """
    dead_units_indices = []
    max_num_clusters = np.max(unit_chart_info[:, 1])  # global max used for setting xaxis.
    num_clusters = np.zeros(max_num_clusters+1)
    cluster_sizes = []
    cluster_peaks = []
    border_cell_indices = []
    place_cells_indices = []
    direction_cell_indices = []
    active_no_type_indices = []

    for unit_index in range(unit_chart_info.shape[0]):
        if unit_chart_info[unit_index, 0] == 0:
            dead_units_indices.append(unit_index)
        else:
            num_clusters[int(unit_chart_info[unit_index, 1])] += 1
            cluster_sizes.extend(unit_chart_info[unit_index, 2])
            cluster_peaks.extend(unit_chart_info[unit_index, 3])

            if unit_chart_info[unit_index, 1] > 0:
                place_cells_indices.append(unit_index)
                is_place_cell = True
            else:
                is_place_cell = False

            if unit_chart_info[unit_index, 6] > 0.47:
                direction_cell_indices.append(unit_index)
                is_direction_cell = True
            else:
                is_direction_cell = False

            if unit_chart_info[unit_index, 9] > 0.5:
                border_cell_indices.append(unit_index)
                is_border_cell = True
            else:
                is_border_cell = False

            if not (is_place_cell or is_direction_cell or is_border_cell):
                active_no_type_indices.append(unit_index)

    # plot
    n_dead_units = len(dead_units_indices)
    n_active_units = unit_chart_info.shape[0] - n_dead_units

    # Collect the indices of units that are all three types
    # (place + border + direction)
    place_border_direction_cells_indices = \
        list(set(place_cells_indices) & set(border_cell_indices) & set(direction_cell_indices))
    
    # Collect the indices of units that are two types (inc. three types)
    # (place + border cells)
    # (place + direction cells)
    # (border + direction cells)
    place_and_border_cells_indices = \
        list(set(place_cells_indices) & set(border_cell_indices))
    place_and_direction_cells_indices = \
        list(set(place_cells_indices) & set(direction_cell_indices))
    border_and_direction_cells_indices = \
        list(set(border_cell_indices) & set(direction_cell_indices))
    
    # Collect the indices of units that are only two types
    # (place  + border - direction),
    # (place  + direction   - border),
    # (border + direction   - place)
    place_and_border_not_direction_cells_indices = \
        list(set(place_and_border_cells_indices) - set(place_border_direction_cells_indices))
    place_and_direction_not_border_cells_indices = \
        list(set(place_and_direction_cells_indices) - set(place_border_direction_cells_indices))
    border_and_direction_not_place_cells_indices = \
        list(set(border_and_direction_cells_indices) - set(place_border_direction_cells_indices))
    
    # Collect the indices of units that are exclusive 
    # place cells, 
    # border cells, 
    # direction cells
    exclusive_place_cells_indices = \
        list(set(place_cells_indices) - (set(place_and_border_cells_indices) | set(place_and_direction_cells_indices)))
    exclusive_border_cells_indices = \
        list(set(border_cell_indices) - (set(place_and_border_cells_indices) | set(border_and_direction_cells_indices)))
    exclusive_direction_cells_indices = \
        list(set(direction_cell_indices) - (set(place_and_direction_cells_indices) | set(border_and_direction_cells_indices)))

    results =  {
        'dead_units_indices': dead_units_indices,
        'place_border_direction_cells_indices': place_border_direction_cells_indices,
        'place_and_border_not_direction_cells_indices': place_and_border_not_direction_cells_indices,
        'place_and_direction_not_border_cells_indices': place_and_direction_not_border_cells_indices,
        'border_and_direction_not_place_cells_indices': border_and_direction_not_place_cells_indices,
        'exclusive_place_cells_indices': exclusive_place_cells_indices,
        'exclusive_border_cells_indices': exclusive_border_cells_indices,
        'exclusive_direction_cells_indices': exclusive_direction_cells_indices,
        'active_no_type_indices': active_no_type_indices,
    }
    
    assert unit_chart_info.shape[0] == sum([len(v) for v in results.values()])

    # Check all values are mutually exclusive
    for key, value in results.items():
        for key2, value2 in results.items():
            if key != key2:
                assert len(set(value) & set(value2)) == 0, f'{key} and {key2} have common elements'

    return results

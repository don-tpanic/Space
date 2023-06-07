import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import logging
import numpy as np

import utils

"""
Given `model_reps` before it is used for training 
either loc_n_rot or border_dist decoding models, 
according to `feature_selection` to lesion columns 
of `model_reps` and return the lesioned model_reps.

The lesion criteria are based on unit chart produced 
by `_single_env_produce_unit_chart.py` in 
`inspect_units.py`. A specific unit chart will have been
saved on disk at `results/.../unit_chart.npy` depending 
on the model and layer.
"""

def lesion(
        config, 
        moving_trajectory, 
        feature_selection,
        model_reps,
        experiment='unit_chart'
    ):
    """
    Lesion `model_reps` according to `feature_selection`
                    
    Args:
        experiment='unit_chart' 
            is always the case because in order to lesion
            we will have done unit charting first.
    """
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
                ]
    
    # load unit chart info
    results_path = utils.load_results_path(
        config=config,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    unit_chart_info = np.load(
        f'{results_path}/unit_chart.npy', allow_pickle=True)
    logging.info(f'unit_chart_info.shape: {unit_chart_info.shape}')

    units_to_lesion_scores = []
    units_to_lesion_indices = []
    if 'gridness' in feature_selection:
        # e.g. For `l2+lesion_borderness_0.37_top_0.1`
        #       we extract the thr=0.37, rank=top, and ratio=0.1
        #       and for units with gridness > 0.37, we lesion 
        #       the top 10% of them. 
        # NOTE: I suppose there is a chance there aren't any qualified
        # but unlikely. In which case, need to adjust thr.
        thr = float(feature_selection.split('_')[2])
        rank = feature_selection.split('_')[3]
        ratio = float(feature_selection.split('_')[4])
        for unit_index in range(unit_chart_info.shape[0]):
            # record units have gridness > thr
            unit_score = unit_chart_info[unit_index, charted_info.index('gridness')]
            if unit_score > thr:
                units_to_lesion_scores.append(unit_score)
                units_to_lesion_indices.append(unit_index)
        
        units_to_lesion_scores = np.array(units_to_lesion_scores)
        units_to_lesion_indices = np.array(units_to_lesion_indices)
                    
        # lesion the top ratio% of units
        if rank == 'top':
            # sort from high to slow
            units_to_lesion_indices = units_to_lesion_indices[np.argsort(units_to_lesion_scores)][::-1]
            num_units_to_lesion = int(len(units_to_lesion_indices) * ratio)
            units_to_lesion_indices = units_to_lesion_indices[num_units_to_lesion:]
        
    elif 'borderness' in feature_selection:
        thr = float(feature_selection.split('_')[2])
        rank = feature_selection.split('_')[3]
        ratio = float(feature_selection.split('_')[4])
        for unit_index in range(unit_chart_info.shape[0]):
            # record units have borderness > thr
            unit_score = unit_chart_info[unit_index, charted_info.index('borderness')]
            if unit_score > thr:
                units_to_lesion_scores.append(unit_score)
                units_to_lesion_indices.append(unit_index)
        
        units_to_lesion_scores = np.array(units_to_lesion_scores)
        units_to_lesion_indices = np.array(units_to_lesion_indices)
        
        # lesion the top ratio% of units
        if rank == 'top':
            # sort from high to slow
            units_to_lesion_indices = units_to_lesion_indices[np.argsort(units_to_lesion_scores)][::-1]
            num_units_to_lesion = int(len(units_to_lesion_indices) * ratio)
            units_to_lesion_indices = units_to_lesion_indices[num_units_to_lesion:]
    
    # lesion based on `units_to_lesion_indices`
    # keep the rest columns
    lesioned_model_reps = np.delete(model_reps, units_to_lesion_indices, axis=1)
    logging.info(
        f'lesioned_model_reps.shape: {lesioned_model_reps.shape}'
    )
    logging.info(
        f'lesioned {(1-lesioned_model_reps.shape[1]/model_reps.shape[1])*100:.3f}% of units'
    )
    return lesioned_model_reps
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import time
import logging
import numpy as np

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
        feature_selection, 
        moving_trajectory, 
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

    units_to_keep = []
    if feature_selection == 'gridness':
        for unit_index in range(unit_chart_info.shape[0]):
            if unit_chart_info[unit_index, charted_info.index('gridness')] > 0.37:
                units_to_keep.append(unit_index)
    
    return model_reps[:, units_to_keep]
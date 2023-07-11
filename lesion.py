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
        reference_experiment,
        decoding_model_choice,
        sampling_rate,
        random_seed,
    ):
    """
    Lesion `model_reps` according to `feature_selection`
                    
    Args:
        experiment='unit_chart' 
            which is when we want to lesion based on unit chart.
            To run this setting, we need to run the corresponding
            unit_chart script first to produce the unit chart,
            which are saved in
                `results/./inspect_units/./unit_chart.npy`
            
        OR 

        experiment='loc_n_rot' or 'border_dist'
            which is when we want to lesion based on coefs
            learned from specific decoding models.
            To run this setting, we need to run the corresponding
            decoding model first to produce the coefs, which are
            saved in 
                `results/./loc_n_rot/./res.npy` or 
                `results/./border_dist/./res.npy`

        NOTE(ken), currently, when experiment != 'unit_chart', loading the 
        coefs require extra info such as sampling rate and random seed as 
        the decoding models are not `general`, unlike the unit chart which
        is produced from `general` outputs. Although, this may be subject to
        change, for example, we might decide that we need to chart units based 
        on their firing conditioned on sampled locations not the entire space.
    """
    logging.info(f'[Check] Initializing lesioning...')
    if reference_experiment == 'unit_chart':
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
            experiment=reference_experiment,
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
                units_to_lesion_indices = units_to_lesion_indices[:num_units_to_lesion]
            
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
                units_to_lesion_indices = units_to_lesion_indices[:num_units_to_lesion]
    
    elif reference_experiment == 'loc_n_rot' or reference_experiment == 'border_dist':
        if 'coef' in feature_selection:
            # NOTE(ken) e.g. For `l2+lesion_coef_thr_top_0.1_loc`
            # the reference experiment uses l2 as feature selection,
            # and we should load this corresponding coefs
            # so we can lesion some of them.
            reference_experiment_feature_selection = \
                feature_selection.split('+')[0]
            logging.info(
                f'feature_selection: {feature_selection}'
            )
            logging.info(
                f'reference_experiment_feature_selection: '\
                f'{reference_experiment_feature_selection}'
            )
            # load task-specific decoding model coefs
            reference_experiment_results_path = \
                utils.load_results_path(
                    config=config,
                    experiment=reference_experiment,
                    feature_selection=reference_experiment_feature_selection,
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

            coef = \
                np.load(
                    f'{reference_experiment_results_path}/res.npy', 
                    allow_pickle=True).item()['coef']  # (n_targets, n_features)
            
            logging.info(f'Loaded coef.shape: {coef.shape}')

            # Due to meeting 24-May-2023, we use absolute
            # values of coef for filtering.
            coef = np.abs(coef)

            if reference_experiment == 'loc_n_rot':
                targets = ['loc', 'rot']  # 'loc' is mean(abs(x) + abs(y))
                # (rob): take the average over x and y columns but keep rot 
                # column as is, so coef \in (2, n_features)
                coef_loc = np.mean(coef[:2, :], axis=0)
                coef_rot = coef[2, :]
                coef = np.vstack((coef_loc, coef_rot))
                logging.info(f'coef_loc.shape: {coef_loc.shape}')
                logging.info(f'coef_rot.shape: {coef_rot.shape}')
                logging.info(f'coef.shape: {coef.shape}')
            else:
                targets = ['border_dist']
                raise NotImplementedError
            
            # ================================
            # apply feature selection criteria
            # e.g. For `l2+lesion_coef_thr_top_0.1_loc`
            #       we extract the thr (dummy), rank=top, and ratio=0.1
            #       lesion the top 10% of them. 
            #       Right now due to loc and rot coef saved together,
            #       we further specify which coef to lesion.
            #       here, _loc means we lesion the coef for loc (x,y averaged)
            rank = feature_selection.split('_')[3]
            ratio = float(feature_selection.split('_')[4])
            target = feature_selection.split('_')[5]
            logging.info(
                f'rank: {rank}, ratio: {ratio}, target: {target}'
            )
            if target == 'loc':
                coef = coef[0, :]
            elif target == 'rot':
                coef = coef[1, :]

            units_to_lesion_scores = coef
            units_to_lesion_indices = np.arange(len(coef))
            logging.info(
                f'units_to_lesion_scores.shape: {units_to_lesion_scores.shape}, '\
                f'units_to_lesion_indices.shape: {units_to_lesion_indices.shape}'
            )
                        
            # lesion the top ratio% of units
            if rank == 'top':
                # sort from high to slow
                units_to_lesion_indices = units_to_lesion_indices[np.argsort(units_to_lesion_scores)][::-1]
                num_units_to_lesion = int(len(units_to_lesion_indices) * ratio)
                units_to_lesion_indices = units_to_lesion_indices[:num_units_to_lesion]

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
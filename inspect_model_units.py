import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import time
import logging
import multiprocessing

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import utils
import data
import models
import unit_metric_computers as umc

"""
Visualize individual model units see 
if there are any patterns (e.g., place cells.)
"""

def _single_model_reps(config):
    """
    Produce model_reps.

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

        return model_reps


def _plot_units_various_ways(
        config,
        filtered_n_units_indices,
        n_units_filtering,
        model_reps,
        model_reps_summed,
        # --------
        sorted_by,
        num_clusters,
        borderness,
        gridness,
        mean_vector_length,
        per_rotation_vector_length,
        coef,
        target_index,
    ):
    """
    A general plotter used by 
        `_single_env_viz_units_ranked_by_coef`
        `_single_env_viz_units_ranked_by_unit_chart`
    where it plots filtered units (based on either 
    coef or a unit chart metric) in a variety of ways.
    
    For now, it plots:
        ratemap | histogram | autocorrelagram | polar plot
    """
    movement_mode=config['movement_mode']
    env_x_min=config['env_x_min']
    env_x_max=config['env_x_max']
    env_y_min=config['env_y_min']
    env_y_max=config['env_y_max']
    multiplier=config['multiplier']

    fig = plt.figure(figsize=(15, 600))
    logging.info(f'[Check] Init plotting..')
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

                # --- subplot1: plot heatmap for the selected units ---
                ax = fig.add_subplot(n_units_filtering, 4, row_index*4+1)
                ax.imshow(heatmap)
                if sorted_by == 'coef':
                    coef_val = f'{coef[target_index, unit_index]:.2f}'
                else:
                    # no coef if sort by any unit chart metrics
                    # as it is task independent.
                    coef_val = 'null'
                ax.set_title(f'u{unit_index}, '\
                                f'coef:{coef_val}, '\
                                f'nfields:{num_clusters[unit_index][0]}, '\
                                f'border:{borderness[unit_index]:.2f}')
                ax.set_xticks([])
                ax.set_yticks([])

                # --- subplot2: plot histogram for the selected units ---
                ax = fig.add_subplot(n_units_filtering, 4, row_index*4+2)
                ax.hist(
                    model_reps_summed[:, rotation, unit_index],
                    bins=10,
                )
                ax.set_xlabel('Activation intensity')
                ax.set_ylabel('Frequency')

                # --- subplot3: plot autocorrelagram for the selected units ---
                _, _, _, _, sac, scorer = \
                            umc._compute_single_heatmap_grid_scores(heatmap)
                
                useful_sac = sac * scorer._plotting_sac_mask
                ax = fig.add_subplot(n_units_filtering, 4, row_index*4+3)
                ax.imshow(useful_sac)
                ax.set_title(f'grid:{gridness[unit_index]:.2f}')
                ax.set_xticks([])
                ax.set_yticks([])

                # --- subplot4: plot polar plot for the selected units ---
                ax = fig.add_subplot(n_units_filtering, 4, row_index*4+4, projection='polar')
                ax.set_title(f'direction:{mean_vector_length[unit_index]:.2f}')
                theta = np.linspace(0, 2*np.pi, model_reps.shape[1], endpoint=False)
                
                ax.plot(theta, per_rotation_vector_length[unit_index])
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_thetagrids([0, 90, 180, 270], labels=['', '', '', ''])
    
    return fig


def _single_env_viz_units_ranked_by_coef_V1(
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
                p_units_filtering = filtering['p_units_filtering']
                filtering_order = filtering['filtering_order']
                # if we filter by percentage, 
                # we overide n_units_filtering
                if p_units_filtering:
                    n_units_filtering = int(coef.shape[1] * p_units_filtering)
                
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
                    ncols=2,
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

                            # # compute fields info and write them on the heatmap
                            # num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
                            #     mean_value_in_clusters, var_value_in_clusters, \
                            #         bounds_heatmap = \
                            #             umc._compute_single_heatmap_fields_info(
                            #                 heatmap=heatmap,
                            #                 pixel_min_threshold=10,
                            #                 pixel_max_threshold=int(heatmap.shape[0]*heatmap.shape[1]*0.5)
                            #             )
                            
                            # plot heatmap on the left column.
                            axes[unit_rank, 0].imshow(heatmap)
                            axes[-1, 0].set_xlabel('Unity x-axis')
                            axes[-1, 0].set_ylabel('Unity z-axis')
                            axes[unit_rank, 0].set_xticks([])
                            axes[unit_rank, 0].set_yticks([])
                            # axes[unit_rank, 0].set_title(
                            #     f'rank{unit_rank},u{unit_index}\n'\
                            #     f'coef{coef[target_index, unit_index]:.1f}'\
                            #     f'{num_clusters},{num_pixels_in_clusters},{max_value_in_clusters} '\
                            # )
                            axes[unit_rank, 0].set_title(f'rank{unit_rank},u{unit_index}')

                            # # plot heatmap contour on the middle column.
                            # axes[unit_rank, 1].imshow(bounds_heatmap)

                            # plot distribution on the right column.
                            axes[unit_rank, 1].hist(
                                model_reps_summed[:, rotation, unit_index],
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
                if p_units_filtering:
                    plt.savefig(
                        f'{figs_path}/units_heatmaps_{targets[target_index]}_'\
                        f'{filtering_order}_{p_units_filtering}_summed.png')
                    logging.info(
                    f'[Saved] units heatmaps {targets[target_index]} {filtering_order} {p_units_filtering}'\
                    f'(summed) to {figs_path}')
                else:
                    plt.savefig(
                        f'{figs_path}/units_heatmaps_{targets[target_index]}_'\
                        f'{filtering_order}_{n_units_filtering}_summed.png')
                    logging.info(
                    f'[Saved] units heatmaps {targets[target_index]} {filtering_order} {n_units_filtering}'\
                    f'(summed) to {figs_path}')
                plt.close()
    else:
        # TODO: metric-based feature selection.
        raise NotImplementedError
    

def _single_env_viz_units_ranked_by_coef_n_save_coef_ranked_unit_charts(
        config_version, 
        experiment,
        reference_experiment,
        feature_selection, 
        decoding_model_choice,
        sampling_rate,
        moving_trajectory,
        random_seed,
        sorted_by='coef',  # dummy, for consistency.
        filterings=[],
    ):
    """
    Plot individual model units in 
    ratemap | autocoorelagram | polar plot.

    Plot the units are sorted by abs(coef).

    Notice the difference to `_single_env_viz_units_ranked_by_unit_chart`
    where in there the units are sorted by a unit_chart metric such as 
    `gridness | borderness | etc.`.

    We separate these two analyses is due to ranked by coef is downstream
    task dependent (i.e. sampling rate, feature selection make a difference.)
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

    # also load unit_chart so we do not repeatedly compute metrics when 
    # plotting unit based on ranked coef.
    # load unit chart info.
    # remember, unit_chart is general so do not supply downstream
    # task specific args such as sampling_rate, feature_selection, etc.
    unit_chart_results_path = utils.load_results_path(
        config=config,
        experiment='unit_chart',
        moving_trajectory=moving_trajectory,
    )
    unit_chart_info = np.load(f'{unit_chart_results_path}/unit_chart.npy', allow_pickle=True)
    num_clusters = unit_chart_info[:, 1]
    num_pixels_in_clusters = unit_chart_info[:, 2]
    max_value_in_clusters = unit_chart_info[:, 3] 
    mean_value_in_clusters = unit_chart_info[:, 4]
    var_value_in_clusters = unit_chart_info[:, 5]
    entire_map_mean = unit_chart_info[:, 6]
    entire_map_var = unit_chart_info[:, 7]
    gridness = unit_chart_info[:, 8]
    borderness = unit_chart_info[:, 9]
    mean_vector_length = unit_chart_info[:, 10]
    per_rotation_vector_length = unit_chart_info[:, 11]
    # filter out dead units, by return unit_index that are active
    # we will then sort based on the active unit_index
    active_unit_indices = np.where(unit_chart_info[:, 0] == 1)[0]
    # ------------------------------------------------------------------------

    # load model outputs
    model_reps = _single_model_reps(config)
    
    if feature_selection in ['l1', 'l2']:
        # load regression coefs as selection criteria
        # for model_reps (per unit)

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

        # metric of interest is the one we sort based on `sorted_by`
        if sorted_by == 'coef':
            metric_of_interest = coef
            # (ken): the shrunk version is wrt active units and we perform 
            # sorting on shrunk version; however, the sorting indices
            # must be mapped back to native model index space so we 
            # extract the correct units.
            metric_of_interest_shrunk = metric_of_interest[:, active_unit_indices]
            logging.info(
                f'metric_of_interest_shrunk.shape: {metric_of_interest_shrunk.shape}'
            )

        for target_index in range(metric_of_interest_shrunk.shape[0]):
            # filter columns of `model_reps` 
            # based on each coef of each target
            # based on `n_units_filtering|p_units_filtering` and `filtering_order`
            for filtering in filterings:
                n_units_filtering = filtering['n_units_filtering']
                p_units_filtering = filtering['p_units_filtering']
                filtering_order = filtering['filtering_order']
                # if we filter by percentage,
                # we overide n_units_filtering
                if p_units_filtering:
                    n_units_filtering = int(metric_of_interest_shrunk.shape[1] * p_units_filtering)

                if filtering_order == 'top_n':
                    filtered_n_units_indices = active_unit_indices[
                        np.argsort(
                            metric_of_interest_shrunk[target_index, :])[::-1][:n_units_filtering]
                    ]

                elif filtering_order == 'mid_n':
                    filtered_n_units_indices = active_unit_indices[
                        np.argsort(
                            metric_of_interest_shrunk[target_index, :])[::-1][
                                int(metric_of_interest_shrunk.shape[1]/2)-int(n_units_filtering/2):
                                int(metric_of_interest_shrunk.shape[1]/2)+int(n_units_filtering/2)]
                    ]
                    
                elif filtering_order == 'random_n':
                    # randomly sample n_units_filtering units
                    # but excluding the top_n (also n_units_filtering)
                    np.random.seed(random_seed)
                    filtered_n_units_indices = active_unit_indices[
                        np.random.choice(
                        np.argsort(
                            metric_of_interest_shrunk[target_index, :])[::-1][n_units_filtering:],
                            n_units_filtering,
                            replace=False)
                    ]
                else:
                    raise NotImplementedError

                # plot summed over rotation heatmap and distribution of loc-wise
                # activation intensities.
                model_reps_summed = np.sum(
                    model_reps, axis=1, keepdims=True)

                # plotter
                # only plot if there aren't too many units,
                # otherwise takes forever..
                if n_units_filtering > 400:
                    plot_various_ways = False
                else:
                    plot_various_ways = True
                if plot_various_ways:
                    fig = _plot_units_various_ways(
                        config=config,
                        filtered_n_units_indices=filtered_n_units_indices,
                        n_units_filtering=n_units_filtering,
                        model_reps=model_reps,
                        model_reps_summed=model_reps_summed,
                        # --------
                        sorted_by=sorted_by,
                        coef=coef,
                        target_index=target_index,
                        num_clusters=num_clusters,
                        borderness=borderness,
                        gridness=gridness,
                        mean_vector_length=mean_vector_length,
                        per_rotation_vector_length=per_rotation_vector_length,
                    )

                    sup_title = f"{filtering_order},{targets[target_index]},"\
                                f"{config['unity_env']},{config['movement_mode']},"\
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
                    plt.tight_layout()
                    plt.suptitle(sup_title)
                    if p_units_filtering:
                        plt.savefig(
                            f'{figs_path}/units_heatmaps_{targets[target_index]}_'\
                            f'{filtering_order}_{p_units_filtering}_summed.png')
                        logging.info(
                        f'[Saved] units heatmaps {targets[target_index]} {filtering_order} {p_units_filtering}'\
                        f'(summed) to {figs_path}')
                    else:
                        plt.savefig(
                            f'{figs_path}/units_heatmaps_{targets[target_index]}_'\
                            f'{filtering_order}_{n_units_filtering}_summed.png')
                        logging.info(
                        f'[Saved] units heatmaps {targets[target_index]} {filtering_order} {n_units_filtering}'\
                        f'(summed) to {figs_path}')
                
                # save the filtered units to disk for further statistical analyses
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

                filtered_unit_chart_info = unit_chart_info[filtered_n_units_indices, :]
                logging.info(
                    f'[Check, before add coef] filtered_unit_chart_info.shape: {filtered_unit_chart_info.shape}'
                )

                # we add extra entry to store coef as later we plot coef against 
                # unit chart metrics.
                filtered_unit_chart_info = np.hstack(
                    ((filtered_unit_chart_info, 
                      coef[target_index, filtered_n_units_indices].T.reshape(-1, 1)
                    ))
                )

                logging.info(
                    f'[Check, after add coef] filtered_unit_chart_info.shape: {filtered_unit_chart_info.shape}'
                )

                if p_units_filtering:
                    np.save(
                        f'{results_path}/unit_chart_{targets[target_index]}_'\
                        f'{filtering_order}_{p_units_filtering}.npy', filtered_unit_chart_info
                    )
                    logging.info(
                        f'[Saved] unit_chart_{targets[target_index]}_{filtering_order}_{p_units_filtering}'\
                        f'to {results_path}'
                    )
                else:
                    np.save(
                        f'{results_path}/unit_chart_{targets[target_index]}_'\
                        f'{filtering_order}_{n_units_filtering}.npy', filtered_unit_chart_info
                    )
                    logging.info(
                        f'[Saved] unit_chart_{targets[target_index]}_{filtering_order}_{n_units_filtering}'\
                        f'to {results_path}'
                    )
    else:
        # metric-based feature selection.
        raise NotImplementedError()


def _single_env_viz_units_by_type_ranked_by_coef(
        config_version, 
        experiment,
        reference_experiment,
        feature_selection, 
        decoding_model_choice,
        sampling_rate,
        moving_trajectory,
        random_seed,
        sorted_by='coef',  # dummy, for consistency.
        filterings=[],
    ):
    """
    `_single_env_viz_units_ranked_by_coef_n_save_coef_ranked_unit_charts`
    produces unit_chart_by_coef where it is a subset of 
    units from the general unit_chart sorted by coef.

    The coef-specific unit_chart has 1 more column than the general unit_chart,
    where the last column is the ranked coef values.

    Here, we can plot units by their types against the coefs.
    e.g. we can plot how the top n coef units' gridness compare 
    to the mid n coef units' gridness.
    """
    unit_type_to_column_index_in_unit_chart = {
        'place_cell': {
            'num_clusters': 1,
            'num_pixels_in_clusters': 2,
            'max_value_in_clusters': 3,
            'mean_value_in_clusters': 4,
            'var_value_in_clusters': 5,
            'entire_map_mean': 6,
            'entire_map_var': 7,
        },
        'grid_cell': {
            'gridness': 8,
        },
        'border_cell': {
            'borderness': 9,
        },
        'direction_cell': {
            'mean_vector_length': 10,
        },
    }

    if reference_experiment == 'loc_n_rot':
        targets = ['loc', 'rot']
    else:
        targets = ['border_dist']

    for unit_type in unit_type_to_column_index_in_unit_chart.keys():
        tracked_metrics = list(
            unit_type_to_column_index_in_unit_chart[unit_type].keys()
        )
        
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
        
        for target_index in range(len(targets)):
            
            if len(tracked_metrics) == 1:
                fig = plt.figure(figsize=(7, 5), dpi=100)
            else:
                fig = plt.figure(figsize=(7*len(tracked_metrics), 5))

            for metric_index, metric in enumerate(tracked_metrics):
                ax = fig.add_subplot(
                    1,                    # nrows
                    len(tracked_metrics), # ncols
                    metric_index+1        # index
                )
                
                for filtering in filterings:
                    n_units_filtering = filtering['n_units_filtering']
                    p_units_filtering = filtering['p_units_filtering']
                    filtering_order = filtering['filtering_order']
                    
                    if p_units_filtering:
                        unit_chart_info = np.load(
                            f'{results_path}/unit_chart_{targets[target_index]}_{filtering_order}_{p_units_filtering}.npy', 
                            allow_pickle=True
                        )
                    else:
                        unit_chart_info = np.load(
                            f'{results_path}/unit_chart_{targets[target_index]}_{filtering_order}_{n_units_filtering}.npy', 
                            allow_pickle=True
                        )

                    # plot distribution of metric and coef based on filtering_order
                    # `type_id` is the column index in unit_chart_info
                    type_id = unit_type_to_column_index_in_unit_chart[unit_type][metric]

                    # NOTE: not every type_id has the same data structure,
                    # e.g. for per unit place fields info, some columns might be ndarray 
                    # such as num_pixels_in_clusters where if the unit has multiple clusters,
                    # the entry for that unit for that column, will be an ndarray.
                    # So for plotting, we need to unpack these arrays.
                    unit_chart_info_unpacked = []
                    for x in unit_chart_info[:, type_id]:
                        if isinstance(x, np.ndarray):
                            unit_chart_info_unpacked.extend(x)
                        else:
                            unit_chart_info_unpacked.append(x)

                    sns.kdeplot(
                        unit_chart_info_unpacked,
                        label=filtering_order,
                        alpha=0.5,
                        ax=ax,
                    )
                    ax.set_xlabel(f'{metric}')
                    ax.set_ylabel(f'pdf')
                    ax.grid()

            sup_title = f"{targets[target_index]},"\
                        f"{config['unity_env']},"\
                        f"{config['model_name']},{feature_selection},"\
                        f"({decoding_model_choice['hparams']}),"\
                        f"{config['output_layer']},"\
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
            plt.suptitle(sup_title)
            if p_units_filtering:
                plt.savefig(
                    f'{figs_path}/{unit_type}_{targets[target_index]}_{p_units_filtering}.png'
                )
            else:
                plt.savefig(
                    f'{figs_path}/{unit_type}_{targets[target_index]}_{n_units_filtering}.png'
                )


def _single_env_viz_units_by_type_pairs_ranked_by_coef(
        config_version, 
        experiment,
        reference_experiment,
        feature_selection, 
        decoding_model_choice,
        sampling_rate,
        moving_trajectory,
        random_seed,
        sorted_by='coef',  # dummy, for consistency.
        filterings=[],
    ):
    """
    `_single_env_viz_units_ranked_by_coef_n_save_coef_ranked_unit_charts`
    produces unit_chart_by_coef where it is a subset of 
    units from the general unit_chart sorted by coef.

    The coef-specific unit_chart has 1 more column than the general unit_chart,
    where the last column is the ranked coef values.

    Here, we can plot units by their types against the coefs.
    e.g. we can plot how the top n coef units' gridness compare 
    to the mid n coef units' gridness.
    """
    unit_type_to_column_index_in_unit_chart = {
        'place_cell': {
            'num_clusters': 1,
            'num_pixels_in_clusters': 2,
            'max_value_in_clusters': 3,
            'mean_value_in_clusters': 4,
            'var_value_in_clusters': 5,
            'entire_map_mean': 6,
            'entire_map_var': 7,
        },
        'grid_cell': {
            'gridness': 8,
        },
        'border_cell': {
            'borderness': 9,
        },
        'direction_cell': {
            'mean_vector_length': 10,
        },
    }

    if reference_experiment == 'loc_n_rot':
        targets = ['loc', 'rot']
    else:
        targets = ['border_dist']

    pairs_to_plot = [
        ('place_cell', 'num_clusters', 'direction_cell', 'mean_vector_length'),
        ('place_cell', 'num_pixels_in_clusters', 'direction_cell', 'mean_vector_length'),
        ('place_cell', 'max_value_in_clusters', 'direction_cell', 'mean_vector_length'),
        ('border_cell', 'borderness', 'direction_cell', 'mean_vector_length')
    ]

    for pair in pairs_to_plot:
        logging.info(f'Plotting {pair}')
        x_unit_type, x_metric, y_unit_type, y_metric = pair
        
        x_column_index = unit_type_to_column_index_in_unit_chart[x_unit_type][x_metric]
        y_column_index = unit_type_to_column_index_in_unit_chart[y_unit_type][y_metric]
        
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
        
        for target_index in range(len(targets)):
            fig = plt.figure(figsize=(7, 5), dpi=100)
            ax = fig.add_subplot(1, 1, 1)

            for filtering in filterings:
                n_units_filtering = filtering['n_units_filtering']
                p_units_filtering = filtering['p_units_filtering']
                filtering_order = filtering['filtering_order']

                if p_units_filtering:
                    unit_chart_info = np.load(
                        f'{results_path}/unit_chart_{targets[target_index]}_{filtering_order}_{p_units_filtering}.npy', 
                        allow_pickle=True
                    )
                else:
                    unit_chart_info = np.load(
                        f'{results_path}/unit_chart_{targets[target_index]}_{filtering_order}_{n_units_filtering}.npy', 
                        allow_pickle=True
                    )

                x_values = unit_chart_info[:, x_column_index]
                y_values = unit_chart_info[:, y_column_index]
                # NOTE(ken), for max_value_in_clusters, we need to unpack
                # the array, but here we can't just unpack but take the max
                # otherwise, we will have different number of x_values and y_values
                # which will cause error in plotting. The maxing shouldnt affect
                # columns of unit_chart_info that are single entries e.g. num_clusters|borderness
                x_values = [np.max(x) for x in x_values]
                y_values = [np.max(y) for y in y_values]

                ax.scatter(
                    x_values, y_values, label=filtering_order, 
                    alpha=0.25, s=2
                )

                ax.set_xlabel(f'{x_metric}')
                ax.set_ylabel(f'{y_metric}')
                ax.grid()

            sup_title = f"{targets[target_index]},"\
                        f"{config['unity_env']},"\
                        f"{config['model_name']},{feature_selection},"\
                        f"({decoding_model_choice['hparams']}),"\
                        f"{config['output_layer']},"\
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
            plt.suptitle(sup_title)

            if p_units_filtering:
                plt.savefig(
                    f'{figs_path}/{x_unit_type}_{x_metric}_vs_{y_unit_type}_{y_metric}_{targets[target_index]}_{p_units_filtering}_scatter.png'
                )
            else:
                plt.savefig(
                    f'{figs_path}/{x_unit_type}_{x_metric}_vs_{y_unit_type}_{y_metric}_{targets[target_index]}_{n_units_filtering}_scatter.png'
                )



def _single_env_produce_unit_chart(
        config_version, 
        experiment,
        moving_trajectory,
        reference_experiment=None,
        feature_selection=None, 
        decoding_model_choice=None,
        sampling_rate=None,
        random_seed=None,
        sorted_by=None,
        filterings=None,  
        # charting all units, use Nones to maintain API consistency
    ):
    """
    Produce unit chart for each unit and save to disk, which 
    will be used for plotting by `_single_env_viz_unit_chart`.

    Unit chart is intended to capture characteristics of each 
    unit (no filtering; ALL units). Currently, the chart includes:
        0. if dead (if true, continue to next unit)
        .  fields info - [
                1. num_clusters, 
                2. num_pixels_in_clusters, 
                3. max_value_in_clusters, 
                4. mean_value_in_clusters, 
                5. var_value_in_clusters,
                6. entire_map_mean,
                7. entire_map_var,
            ]
        8. gridness - gridness score
        9. borderness - border score
        .  directioness - [
                10. mean_vector_length,
                11. per_rotation_vector_length,
            ]
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
                    # --- fields info
                    'num_clusters', 
                    'num_pixels_in_clusters', 
                    'max_value_in_clusters', 
                    'mean_value_in_clusters', 
                    'var_value_in_clusters',
                    'entire_map_mean',
                    'entire_map_var',
                    # ---
                    'gridness',
                    # ---
                    'borderness',
                    # --- directioness
                    'mean_vector_length',
                    'per_rotation_vector_length',
                    # ---
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
        logging.info(f'[Charting] unit_index: {unit_index}')

        if movement_mode == '2d':
            # reshape to (n_locations, n_rotations, n_features)
            # but rotation dimension is 1 after summing, so 
            # we just hard code 0.
            heatmap = model_reps_summed[:, 0, unit_index].reshape(
                (env_x_max*multiplier-env_x_min*multiplier+1, 
                env_y_max*multiplier-env_y_min*multiplier+1) )
            # rotate heatmap to match Unity coordinate system
            # ref: tests/testReshape_forHeatMap.py
            heatmap = np.rot90(heatmap, k=1, axes=(0, 1))

            ###### Go thru each required info, maybe modularise later.
            if umc._is_dead_unit(heatmap):
                logging.info(f'Unit {unit_index} dead.')
                unit_chart_info[unit_index, 0] = np.array([0])
                continue
            else:
                logging.info(f'Unit {unit_index} active')
                unit_chart_info[unit_index, 0] = np.array([1])
                # compute, collect and save unit chart info
                # 1. fields info
                num_clusters, num_pixels_in_clusters, max_value_in_clusters, \
                    mean_value_in_clusters, var_value_in_clusters, \
                        bounds_heatmap = \
                            umc._compute_single_heatmap_fields_info(
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
                            umc._compute_single_heatmap_grid_scores(heatmap)
                unit_chart_info[unit_index, 8] = score_60_

                # 3. borderness
                border_score = umc._compute_single_heatmap_border_scores(heatmap)
                unit_chart_info[unit_index, 9] = border_score

                # 4. directioness (use model_reps instead of model_reps_summed)
                directional_score, per_rotation_vector_length = \
                    umc._compute_single_heatmap_directional_scores(
                        activation_maps=model_reps[:, :, unit_index]
                    )
                unit_chart_info[unit_index, 10] = directional_score
                unit_chart_info[unit_index, 11] = per_rotation_vector_length
        
    results_path = utils.load_results_path(
        config=config,
        experiment=experiment,
        moving_trajectory=moving_trajectory,
    )
    fpath = f'{results_path}/unit_chart.npy'
    np.save(fpath, unit_chart_info)
    logging.info(f'[Saved] {fpath}')


def _single_env_viz_units_ranked_by_unit_chart(
        config_version, 
        experiment,
        moving_trajectory,
        reference_experiment=None,
        feature_selection=None, 
        decoding_model_choice=None,
        sampling_rate=None,
        random_seed=None,
        sorted_by='gridness',
        filterings=[]
    ):
    """
    Based on unit chart info produced by `_single_env_produce_unit_chart`,
    We can load the chart and sort by `gridness | border | directioness | etc.` 
    and visualize the top_n, mid_n sorted units in terms of 
    `ratemaps | autocorrelagrams | polar plots`.

    Notice the postfix `sorted_from_unit_chart` is to distinguish from 
    `*viz_fields_info*` and `*viz_units*` which are based on coef ranking 
    which are based on specific combination of feature selection, sampling rate,
    etc. Whereas here the visualization is general to all the settings and are
    not coef-based but gridness-based (i.e. filtering is based on gridness).
    """
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{TF_NUM_INTRAOP_THREADS}"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    config = utils.load_config(config_version)
    
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

    # TODO: potentially we should first select based on num_clusters >= 4+
    # and then sort by gridness.

    num_clusters = unit_chart_info[:, 1]
    gridness = unit_chart_info[:, 8]
    borderness = unit_chart_info[:, 9]
    mean_vector_length = unit_chart_info[:, 10]
    per_rotation_vector_length = unit_chart_info[:, 11]
    
    # filter out dead units, by return unit_index that are active
    # we will then sort based on the active unit_index
    active_unit_indices = np.where(unit_chart_info[:, 0] == 1)[0]

    # metric of interest is the one we sort based on `sorted_by`
    if sorted_by == 'num_clusters':
        metric_of_interest = num_clusters
        # (ken): the shrunk version is wrt active units and we perform 
        # sorting on shrunk version; however, the sorting indices
        # must be mapped back to native model index space so we 
        # extract the correct units.
        metric_of_interest_shrunk = metric_of_interest[active_unit_indices]

    elif sorted_by == 'gridness':
        metric_of_interest = gridness
        metric_of_interest_shrunk = metric_of_interest[active_unit_indices]

    elif sorted_by == 'borderness':
        metric_of_interest = borderness
        metric_of_interest_shrunk = metric_of_interest[active_unit_indices]
    
    elif sorted_by == 'directioness':
        metric_of_interest = mean_vector_length
        metric_of_interest_shrunk = metric_of_interest[active_unit_indices]
    
    # visualize top_n, mid_n, random_n units' gridness
    for filtering in filterings:
        n_units_filtering = filtering['n_units_filtering']
        p_units_filtering = filtering['p_units_filtering']
        filtering_order = filtering['filtering_order']
        # if we filter by percentage,
        # we overide n_units_filtering
        if p_units_filtering:
            n_units_filtering = int(metric_of_interest_shrunk.shape[0] * p_units_filtering)

        logging.info(f'metric_of_interest.shape: {metric_of_interest.shape}')
        logging.info(f'metric_of_interest_shrunk.shape: {metric_of_interest_shrunk.shape}')
        logging.info(f'filtering_order: {filtering_order}')
        logging.info(f'n_units_filtering: {n_units_filtering}')

        if filtering_order == 'top_n':
            filtered_n_units_indices = active_unit_indices[
                np.argsort(metric_of_interest_shrunk)[::-1][:n_units_filtering]
            ]   

        elif filtering_order == 'mid_n':
            filtered_n_units_indices = active_unit_indices[
                np.argsort(metric_of_interest_shrunk)[::-1][
                    int(metric_of_interest_shrunk.shape[0]/2)-int(n_units_filtering/2):
                    int(metric_of_interest_shrunk.shape[0]/2)+int(n_units_filtering/2)]
            ]
            
        elif filtering_order == 'random_n':
            # randomly sample n_units_filtering units
            # but excluding the top_n (also n_units_filtering)
            np.random.seed(random_seed)
            filtered_n_units_indices = active_unit_indices[
                np.random.choice(
                np.argsort(metric_of_interest_shrunk)[::-1][n_units_filtering:],
                    n_units_filtering,
                    replace=False)
            ]
        else:
            raise NotImplementedError

        # plotter
        fig = _plot_units_various_ways(
            config,
            filtered_n_units_indices,
            n_units_filtering,
            model_reps,
            model_reps_summed,
            # --------
            sorted_by,
            num_clusters,
            borderness,
            gridness,
            mean_vector_length,
            per_rotation_vector_length,
            # --------
            coef=None,          # not applicable
            target_index=None,  # not applicable

        )

        figs_path = utils.load_figs_path(
            config=config,
            experiment=experiment,
            moving_trajectory=moving_trajectory,
        )
        plt.tight_layout()
        if p_units_filtering:
            plt.savefig(
                f'{figs_path}/{sorted_by}_sorted_from_unit_chart_{filtering_order}_{p_units_filtering}.png'
            )
            logging.info(
                f'[Saved] {figs_path}/{sorted_by}_sorted_from_unit_chart_{filtering_order}_{p_units_filtering}.png'
            )
        else:
            plt.savefig(
                f'{figs_path}/{sorted_by}_sorted_from_unit_chart_{filtering_order}_{n_units_filtering}.png'
            )
            logging.info(
                f'[Saved] {figs_path}/{sorted_by}_sorted_from_unit_chart_{filtering_order}_{n_units_filtering}.png'
            )
        plt.close()
    

def _single_env_viz_unit_chart(
        config_version, 
        experiment,
        moving_trajectory,
        reference_experiment=None,
        feature_selection=None, 
        decoding_model_choice=None,
        sampling_rate=None,
        random_seed=None,
        sorted_by=None,
        filterings=None,
    ):
    """
    Visualize unit chart info produced by `_single_env_produce_unit_chart`.
    """
    charted_info = [
                    'dead',
                    'num_clusters', 
                    'num_pixels_in_clusters', 
                    'max_value_in_clusters', 
                    'gridness',
                    'borderness',
                    'directioness',
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
    #   For `% borderness`, we iterate through unit_chart_info and count
    # the borderness of qualified units (w borderness>0.5).
    dead_units = 0
    max_num_clusters = np.max(unit_chart_info[:, 1])  # global max used for setting xaxis.
    num_clusters = np.zeros(max_num_clusters+1)
    cluster_sizes = []
    cluster_peaks = []
    grid_cells = 0
    border_cells = 0

    for unit_index in range(unit_chart_info.shape[0]):
        if unit_chart_info[unit_index, 0] == 0:
            dead_units += 1
        else:
            num_clusters[int(unit_chart_info[unit_index, 1])] += 1
            cluster_sizes.extend(unit_chart_info[unit_index, 2])
            cluster_peaks.extend(unit_chart_info[unit_index, 3])
            if unit_chart_info[unit_index, 8] > 0.37:
                grid_cells += 1
            if unit_chart_info[unit_index, 9] > 0.5:
                border_cells += 1

    # plot
    fig, axes = plt.subplots(
        nrows=len(charted_info), 
        ncols=1, 
        figsize=(5, 5*len(charted_info))
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
    axes[0].grid()

    # 1-each bar is % of a num_clusters
    axes[1].bar(
        np.arange(max_num_clusters+1),
        num_clusters/unit_chart_info.shape[0]
    )
    axes[1].set_xlabel('num_clusters')
    axes[1].set_ylabel('% units')
    axes[1].set_title(f'% units with 1, 2, .., {max_num_clusters[0]} clusters')
    axes[1].set_ylim([-.05, 1.05])
    axes[1].grid()

    # 2-each bar is % of a cluster size (bined)
    axes[2].hist(
        cluster_sizes, bins=20, density=True
    )
    axes[2].set_xlabel('cluster size')
    axes[2].set_ylabel('density')
    axes[2].set_title(f'cluster size distribution')
    axes[2].grid()

    # 3-each bar is % of a cluster peak (bined)
    axes[3].hist(
        cluster_peaks, bins=20, density=True
    )
    axes[3].set_xlabel('cluster peak')
    axes[3].set_ylabel('density')
    axes[3].set_title(f'cluster peak distribution')
    axes[3].grid()

    # 4-each bar is % of a gridness
    # non-grid left, grid right
    axes[4].bar(
        np.arange(2),
        [1-grid_cells/unit_chart_info.shape[0],
            grid_cells/unit_chart_info.shape[0]],
        color=['grey', 'blue']
    )
    axes[4].set_xticks(np.arange(2))
    axes[4].set_xticklabels(['non-grid', 'grid'])
    axes[4].set_ylabel('% units')
    axes[4].set_title(f'% units grid/non-grid')
    axes[4].set_ylim([-.05, 1.05])
    axes[4].grid()

    # 5-each bar is % of a borderness
    # non-border left, border right
    axes[5].bar(
        np.arange(2),
        [1-border_cells/unit_chart_info.shape[0],
            border_cells/unit_chart_info.shape[0]],
        color=['grey', 'blue']
    )
    axes[5].set_xticks(np.arange(2))
    axes[5].set_xticklabels(['non-border', 'border'])
    axes[5].set_ylabel('% units')
    axes[5].set_title(f'% units border/non-border')
    axes[5].set_ylim([-.05, 1.05])
    axes[5].grid()

    # 6-each bar is % of a directioness
    axes[6].hist(
        unit_chart_info[:, 10], bins=20, density=True
    )
    axes[6].set_xlabel('directioness')
    axes[6].set_ylabel('density')
    axes[6].set_title(f'directioness distribution')
    axes[6].grid()
    
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
        sorted_by,
        filterings,
    ):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with multiprocessing.Pool(processes=CPU_NUM_PROCESSES) as pool:
        for model_name in model_names:
            envs_dict = data.load_envs_dict(model_name, envs)
            config_versions=list(envs_dict.keys())
            for config_version in config_versions:
                for moving_trajectory in moving_trajectories:
                    if experiment == 'unit_chart':
                        res = pool.apply_async(
                            target_func,
                            args=(
                                config_version, 
                                experiment,
                                moving_trajectory,
                                sorted_by,
                                filterings,
                            )
                        )
                    else:
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
                                                sorted_by,
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
        sorted_by,
        filterings,
        cuda_id_list=[0, 1, 2, 3, 4, 5, 6, 7],
    ):
    args_list = []
    for model_name in model_names:
        envs_dict = data.load_envs_dict(model_name, envs)
        config_versions=list(envs_dict.keys())
        # args_list = []
        for config_version in config_versions:
            for moving_trajectory in moving_trajectories:
                if experiment == 'unit_chart':
                    single_entry = {}
                    single_entry['config_version'] = config_version
                    single_entry['experiment'] = experiment
                    single_entry['moving_trajectory'] = moving_trajectory
                    single_entry['sorted_by'] = sorted_by
                    single_entry['filterings'] = filterings
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
    experiment = 'unit_chart_by_coef'
    reference_experiment = 'loc_n_rot'
    envs = ['env28_r24']
    movement_modes = ['2d']
    sampling_rates = [0.3]
    random_seeds = [42]
    model_names = ['vgg16']
    moving_trajectories = ['uniform']
    decoding_model_choices = [{'name': 'ridge_regression', 'hparams': 1.0}]
    feature_selections = ['l2']
    sorted_by = 'coef'
    filterings = [
        {'filtering_order': 'top_n', 'n_units_filtering': None, 'p_units_filtering': 0.1},
        {'filtering_order': 'random_n', 'n_units_filtering': None, 'p_units_filtering': 0.1},
        {'filtering_order': 'mid_n', 'n_units_filtering': None, 'p_units_filtering': 0.1},
    ]
    # ======================================== #
    ###  How to run ###
    # 1. run `_single_env_produce_unit_chart` to produce downstream task independent 
    #       unit chart for each unit and save to disk.
    #       set `experiment='unit_chart'` and `reference_experiment=None`
    # 2a. run `_single_env_viz_units_ranked_by_coef_n_save_coef_ranked_unit_charts` to
    #      produce task depedent unit chart filtered by coef.
    #      set `experiment='unit_chart_by_coef'` and `reference_experiment='loc_n_rot|border_dist'`
    # 2b. run `_single_env_viz_units_by_type_ranked_by_coef` to plot unit_chart metric against coef.
    #      set `experiment='unit_chart_by_coef'` and `reference_experiment='loc_n_rot|border_dist'`
    # 3. run `_single_env_viz_unit_chart` to plot aggregate unit chart info.
    #       set `experiment='unit_chart'` and `reference_experiment=None`
    # 4. run `_single_env_viz_units_ranked_by_unit_chart` to plot unit chart info ranked by a unit_chart metric.
    #       set `experiment='unit_chart'` and `reference_experiment=None`
    # NOTE: 2a, 2b, 3, 4 all require 1 to be run first.
    # NOTE: 2b requires 2a
    # NOTE: 3, 4 less interesting than 2b which compares unit chart info againt coef.

    multi_envs_inspect_units_GPU(
    # multi_envs_inspect_units_CPU(
        # target_func=_single_env_produce_unit_chart,                       # set experiment='unit_chart'
        # target_func=_single_env_viz_units_ranked_by_unit_chart,           # set experiment='unit_chart'
        # target_func=_single_env_viz_unit_chart,                           # set experiment='unit_chart'
        # target_func=_single_env_viz_units_ranked_by_coef_n_save_coef_ranked_unit_charts,    # set experiment='unit_chart_by_coef'
        # target_func=_single_env_viz_units_by_type_ranked_by_coef,
        target_func=_single_env_viz_units_by_type_pairs_ranked_by_coef,
        envs=envs,
        model_names=model_names,
        experiment=experiment,
        reference_experiment=reference_experiment,
        moving_trajectories=moving_trajectories,
        sampling_rates=sampling_rates,
        feature_selections=feature_selections,
        decoding_model_choices=decoding_model_choices,
        random_seeds=random_seeds,
        sorted_by=sorted_by,
        filterings=filterings,
        cuda_id_list=[0, 1, 2, 3, 4, 5, 6, 7],
    )

    # print time elapsed
    logging.info(f'Time elapsed: {time.time() - start_time:.2f}s')
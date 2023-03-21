import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import multiprocessing
import numpy as np

import utils
import evaluations
from models import load_model
from data import load_data
import dimension_reduction


def single_config_reps(config):
    unity_env = config['unity_env']
    model_name = config['model_name']
    output_layer = config['output_layer']
    n_components = config['n_components']
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
    
    # use raw image input 
    if model_name == 'none':
        model_reps = preprocessed_data.reshape(preprocessed_data.shape[0], -1)
        print(f'[Check] raw image input shape: {model_reps.shape}')
    # use model output
    else:
        # (n, 4096)
        model_reps = model.predict(preprocessed_data)
        if len(model_reps.shape) > 2:
            # when not a fc layer, we need to flatten the output dim
            # except the batch dim.
            model_reps = model_reps.reshape(model_reps.shape[0], -1)
        print(f'[Check] model_reps.shape: {model_reps.shape}')

    # (n_samples, n_components)
    # keep all components due to we want to see the explained variance ratio
    # in case of PCA; if not PCA, second output returns None.
    components, stats, fitter = \
        dimension_reduction.compute_components(
            model_reps, 
            reduction_method=config['reduction_method'],
            reduction_hparams=reduction_hparams,
        )
    print(f'[Check] components produced, fitter returned')
    return components, stats, fitter, results_path


def execute(config_version):
    config = utils.load_config(config_version)
    components, stats, fitter, results_path = single_config_reps(config)

    # heatmaps
    print(f'[Analysis] Running heamap analysis')
    # the agent takes n_rotations views of the same position
    # each view is considered an independent data-point to dimension reduction
    # instead of in the case of concatenation where all frames is considered 
    # one data-point.
    # So when save components for analysis, we need to save component 
    # by location and by rotation.
    components = components.reshape(
        (components.shape[0] // config['n_rotations'],  # n_locations
        config['n_rotations'],                          # n_rotations
        components.shape[1])                            # all components
    )
    print(f'[Check] components.shape: {components.shape}')

    # save the top n components for evaluations
    # note save as a matrix of (n_locations, n_rotations) per component
    for i in range(config['n_components']):
        np.save(
            f'{results_path}/components_{i+1}.npy', 
            components[:, :, i]
        )

    if config['reduction_method'] == 'pca':
        # save the explained variance ratio
        np.save(
            f'{results_path}/explained_variance_ratio.npy', 
            stats
        )
        evaluations.plot_variance_explained(
            movement_mode=config['movement_mode'],
            model_name=config['model_name'],
            output_layer=config['output_layer'],
            reduction_method=config['reduction_method'],
            results_path=results_path,
        )

    # eval
    evaluations.plot_components(
        unity_env=config['unity_env'],
        n_components=config['n_components'],
        n_rotations=config['n_rotations'],
        movement_mode=config['movement_mode'],
        model_name=config['model_name'],
        output_layer=config['output_layer'],
        reduction_method=config['reduction_method'],
        results_path=results_path,
        x_min=config['x_min'],
        x_max=config['x_max'],
        y_min=config['y_min'],
        y_max=config['y_max'],
        multiplier=config['multiplier'],
    )


def multiproc_execute():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "2"  # limit each execute's max threads
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"  # limit each execute's max threads

    num_processes = 20
    envs = ['28_r24', '29_r24', '30_r24', '31_r24', '32_r24', '33_r24']
    movement_modes = ['2d']
    dim_reductions = ['pca', 'nmf', 'maxvar']
    n_components_list = [9]
    model_types_n_reps = {'vgg16': 'fc2'}

    with multiprocessing.Pool(num_processes) as pool:
        for env in envs:
            for movement_mode in movement_modes:
                for dim_reduction in dim_reductions:
                    for n_components in n_components_list:
                        for model_type, model_rep in model_types_n_reps.items():
                            config_version = \
                                f'env{env}_{movement_mode}_{model_type}_{model_rep}_' \
                                f'{n_components}_{dim_reduction}'

                            results = pool.apply_async(
                                execute, 
                                args=[config_version]
                            )
                
        print(results.get())
        pool.close()
        pool.join()


def multicuda_execute(target_func, cuda_id_list=[0, 1, 2, 3, 4, 5, 6, 7]):
    envs = ['28_r24', '29_r24', '30_r24', '31_r24', '32_r24', '33_r24']
    movement_modes = ['2d']
    dim_reductions = ['pca', 'nmf', 'maxvar']
    n_components_list = [9]
    model_types_n_reps = {'vgg16': 'fc2'}

    config_versions = []
    for env in envs:
        for movement_mode in movement_modes:
            for dim_reduction in dim_reductions:
                for n_components in n_components_list:
                    for model_type, model_rep in model_types_n_reps.items():
                        config_version = \
                            f'env{env}_{movement_mode}_{model_type}_{model_rep}_' \
                            f'{n_components}_{dim_reduction}'
                        config_versions.append(config_version)

    args_list = []
    for config_version in config_versions:
        single_entry = {}
        single_entry['config_version'] = config_version
        args_list.append(single_entry)

    print(args_list)
    print(len(args_list))
    utils.cuda_manager(
        target_func, args_list, cuda_id_list
    )


if __name__ == "__main__":
    import time
    start_time = time.time()

    # multiproc_execute()
    multicuda_execute(execute)

    end_time = time.time()
    time_elapsed = (end_time - start_time) / 3600
    print(f'Time elapsed: {time_elapsed} hrs')
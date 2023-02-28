import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import multiprocessing
import utils
from models import load_model
from data import load_data
import dimension_reduction
import heatmap_analysis
import component_analysis


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
    heatmap_analysis.execute(
        components=components,
        stats=stats,
        config=config,
        results_path=results_path,
    )

    if config['model_name'] == 'none':
        # component reconstructions for now 
        # only supports raw image reps
        component_analysis.execute(
            fitter=fitter,
            config=config, 
            results_path=results_path,
        )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    execute('env13_2d_none_raw_9_nmf')

    # num_processes = 70
    # envs = [13]
    # movement_modes = ['2d']
    # dim_reductions = ['pca', 'nmf']
    # n_components_list = [9]
    # model_types_n_reps = {'vgg16': 'fc2', 'none': 'raw'}

    # with multiprocessing.Pool(num_processes) as pool:
    #     for env in envs:
    #         for movement_mode in movement_modes:
    #             for dim_reduction in dim_reductions:
    #                 for n_components in n_components_list:
    #                     for model_type, model_rep in model_types_n_reps.items():
    #                         config_version = \
    #                             f'env{env}_{movement_mode}_{model_type}_{model_rep}_' \
    #                             f'{n_components}_{dim_reduction}'

    #                         results = pool.apply_async(
    #                             execute, 
    #                             args=[config_version]
    #                         )
                
    #     print(results.get())
    #     pool.close()
    #     pool.join()
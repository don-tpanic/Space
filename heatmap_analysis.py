import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
from models import load_model
from data import load_data
import dimension_reduction
import evaluations
import utils


def execute(config_version):
    # load config
    config = utils.load_config(config_version)
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
        print(f'raw image input shape: {model_reps.shape}')
    # use model output
    else:
        # (n, 4096)
        model_reps = model.predict(preprocessed_data)
        if len(model_reps.shape) > 2:
            # when not a fc layer, we need to flatten the output dim
            # except the batch dim.
            model_reps = model_reps.reshape(model_reps.shape[0], -1)
        print(f'model_reps.shape: {model_reps.shape}')

    # (n_samples, n_components)
    # keep all components due to we want to see the explained variance ratio
    # in case of PCA; if NMF, second output is None.
    components, explained_variance_ratio, _ = \
        dimension_reduction.compute_components(
            model_reps, 
            reduction_method=reduction_method,
            reduction_hparams=reduction_hparams,
        )
    print(f'components.shape: {components.shape}')

    # the agent takes n_rotations views of the same position
    # each view is considered an independent data-point to dimension reduction
    # instead of in the case of concatenation where all frames is considered 
    # one data-point.
    # So when save components for analysis, we need to save component 
    # by location and by rotation.
    components = components.reshape(
        (components.shape[0] // n_rotations,  # n_locations
        n_rotations,                          # n_rotations
        components.shape[1])                  # all components
    )
    print(f'components.shape: {components.shape}')

    # save the top n components for evaluations
    # note save as a matrix of (n_locations, n_rotations) per component
    for i in range(n_components):
        np.save(
            f'{results_path}/components_{i+1}.npy', 
            components[:, :, i]
        )

    if reduction_method == 'pca':
        # save the explained variance ratio
        np.save(
            f'{results_path}/explained_variance_ratio.npy', 
            explained_variance_ratio
        )
        evaluations.plot_variance_explained(
            movement_mode=movement_mode,
            model_name=model_name,
            output_layer=output_layer,
            reduction_method=reduction_method,
            results_path=results_path,
        )

    # eval
    evaluations.plot_components(
        unity_env=unity_env,
        n_components=n_components,
        n_rotations=n_rotations,
        movement_mode=movement_mode,
        model_name=model_name,
        output_layer=output_layer,
        reduction_method=reduction_method,
        results_path=results_path,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        multiplier=multiplier,
    )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_version = 'env3_2d_vgg16_fc2_9_nmf'
    execute(config_version)

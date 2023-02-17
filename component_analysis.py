import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import matplotlib.pyplot as plt
from models import load_model
from data import load_data
import dimension_reduction
import evaluations
import utils


def execute(config_version, component_matrix_type):
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
    results_path = f'results/{unity_env}/{movement_mode}/{model_name}/{output_layer}/{reduction_method}'
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
        print(f'model_reps.shape: {model_reps.shape}')

    if movement_mode == '2d':
        # due to each position in 2d takes 6 views, we 
        # concat them so each position has 1 vector.
        # also we use the fact the file names are sorted so
        # every `n_rotations` files are the same position
        n_rows = int(model_reps.shape[0] / n_rotations)
        n_cols = int(model_reps.shape[1] * n_rotations)
        model_reps = model_reps.reshape((n_rows, n_cols))
        print(f'model_reps.shape: {model_reps.shape}')

    # (n_samples, n_components)
    # keep all components due to we want to see the explained variance ratio
    # in case of PCA; if NMF, second output is None.
    component_matrix = dimension_reduction.compute_component_matrix(
        model_reps, 
        reduction_method=reduction_method, 
        component_matrix_type=component_matrix_type
    )
    
    # reshape each principle axis back to image shape
    fig, ax = plt.subplots(n_components, n_rotations, figsize=(20, 20))
    for i in range(n_components):
        for r in range(n_rotations):
            step_size = int(component_matrix.shape[1] / n_rotations)
            component_matrix_as_img_per_rot = \
                component_matrix[i, r*step_size:(r+1)*step_size].reshape((224, 224, 3))
            
            # # zero out the values that are less than 0
            # component_matrix_as_img_per_rot[component_matrix_as_img_per_rot < 0] = 0
            # normalize [0, 1]
            component_matrix_as_img_per_rot = \
                (component_matrix_as_img_per_rot - np.min(component_matrix_as_img_per_rot)) / \
                (np.max(component_matrix_as_img_per_rot) - np.min(component_matrix_as_img_per_rot))     # TODO: maybe should norm globally?
    
            ax[i, r].imshow(component_matrix_as_img_per_rot)
            ax[i, r].set_title(f'comp. {i+1}, rot. {r+1}')
            ax[i, r].axis('off')

    plt.tight_layout()
    plt.savefig(f'{results_path}/{component_matrix_type}_as_img.png')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_version = 'env3_2d_none_raw_9_pca'
    execute(config_version, component_matrix_type='loadings')

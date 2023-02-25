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

    component_matrix = dimension_reduction.compute_component_matrix(
        model_reps, 
        reduction_method=reduction_method, 
        component_matrix_type=component_matrix_type
    )
    
    # reshape each principle axis back to image shape
    fig, ax = plt.subplots(n_components, 1, figsize=(20, 20))

    # normalize and plot
    for i in range(n_components):
        print(f'component {i}')
        component_matrix_i = component_matrix[i, :].reshape((224, 224, 3))
        
        # normalize [0, 1]
        max_val = np.max(component_matrix_i)
        min_val = np.min(component_matrix_i)
        print(f'max_val: {max_val}, min_val: {min_val}')
        component_matrix_i = (component_matrix_i - min_val) / (max_val - min_val)
        ax[i].imshow(component_matrix_i)
        ax[i].set_title(f'comp.{i+1}')
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{results_path}/{component_matrix_type}_as_img.png')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_version = 'env9_2d_none_raw_9_pca'
    execute(config_version, component_matrix_type='loadings')
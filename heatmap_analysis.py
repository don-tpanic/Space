import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
from models import load_model
from data import load_data
import dimension_reduction
import evaluations
import utils


def execute(components, stats, config, results_path):
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


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

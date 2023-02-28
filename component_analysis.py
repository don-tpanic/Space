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


def execute(fitter, config, results_path, component_matrix_type='loadings'):
    print(f'[Analysis] Running component analysis')

    if config['reduction_method'] == 'pca':
        Sigma = fitter.singular_values_
        Vt = fitter.components_
        if component_matrix_type == 'Vt':
            component_matrix = Vt
        elif component_matrix_type == 'loadings':
            loadings = np.dot(np.diag(Sigma), Vt)
            component_matrix = loadings
    
    elif config['reduction_method'] == 'nmf':
        print(f'running NMF...')
        print('Warning: NMF loadings are not defined and somewhat questionable.')
        print('Warning: NMF loadings are not defined and somewhat questionable.')
        print('Warning: NMF loadings are not defined and somewhat questionable.')
        component_matrix = fitter.components_
    
    elif config['reduction_method'] == 'ica':
        print(f'running ICA...')
        print('Warning: ICA loadings are not defined and somewhat questionable.')
        print('Warning: ICA loadings are not defined and somewhat questionable.')
        print('Warning: ICA loadings are not defined and somewhat questionable.')
        component_matrix = fitter.components_
    
    # reshape each principle axis back to image shape
    fig, ax = plt.subplots(config['n_components'], 1, figsize=(20, 20))

    # normalize and plot
    for i in range(config['n_components']):
        component_matrix_i = component_matrix[i, :].reshape((224, 224, 3))
        # normalize [0, 1]
        max_val = np.max(component_matrix_i)
        min_val = np.min(component_matrix_i)
        # print(f'component {i}, max_val: {max_val}, min_val: {min_val}')
        component_matrix_i = (component_matrix_i - min_val) / (max_val - min_val)
        ax[i].imshow(component_matrix_i)
        ax[i].set_title(f'comp.{i+1}')
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{results_path}/{component_matrix_type}_as_img.png')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
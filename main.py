import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
from models import load_model
from data import load_data
import dimension_reduction
import evaluations


def execute(
        model_name, 
        output_layer, 
        n_components, 
        movement_mode, 
        data_path, 
        reduction_method,
    ):
    results_path = f'results/{movement_mode}/{model_name}/{output_layer}/{reduction_method}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if model_name == 'none':
        preprocess_func = None
    else:
        model, preprocess_func = load_model(model_name, output_layer)

    preprocessed_data = load_data(
        data_path, movement_mode, preprocess_func)
    
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
        # every 6 files are the same position  # TODO: is there a more elegant way?
        n_rows = int(model_reps.shape[0] / 6)
        n_cols = int(model_reps.shape[1] * 6)
        model_reps = model_reps.reshape((n_rows, n_cols))
        print(f'model_reps.shape: {model_reps.shape}')

    # (n_samples, n_components)
    # keep all components due to we want to see the explained variance ratio
    # in case of PCA; if NMF, second output is None.
    components, explained_variance_ratio = \
        dimension_reduction.compute_components(
            model_reps, reduction_method=reduction_method
        )

    # save the top n components for evaluations
    for i in range(n_components):
        np.save(
            f'{results_path}/components_{i+1}.npy', 
            components[:, i]
        )

    if reduction_method == 'pca':
        # save the explained variance ratio
        np.save(
            f'{results_path}/explained_variance_ratio.npy', 
            explained_variance_ratio
        )
        evaluations.plot_variance_explained(results_path, reduction_method)

    # eval
    evaluations.plot_components(
        n_components,
        movement_mode,
        model_name,
        output_layer,
        reduction_method,
    )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    movement_mode = '1d'
    model_name = 'vgg16'
    output_layer = 'fc2'
    n_components = 9
    data_path = f'data/unity'
    reduction_method = 'nmf'

    execute(
        model_name, 
        output_layer, 
        n_components, 
        movement_mode, 
        data_path,
        reduction_method,
    )

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image \
    import load_img, img_to_array


def generate_random_data(
        data_path,
        movement_mode,
        x_min,
        x_max,
        y_min,
        y_max,
        multiplier,
        n_rotations,
        random_seed,
        image_shape=(224, 224, 3),
    ):
    """Generate random data for testing purposes"""

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # generate random data
    np.random.seed(random_seed)

    if movement_mode == '1d':
        n_stops_x = len( range(x_min*multiplier, x_max*multiplier+1) )
        n_samples = n_stops_x
    elif movement_mode == '2d':
        n_stops_x = len( range(x_min*multiplier, x_max*multiplier+1) )
        n_stops_y = len( range(y_min*multiplier, y_max*multiplier+1) )
        n_samples = (n_stops_x * n_stops_y) * n_rotations
    print(f'n_samples: {n_samples}')

    batch_x = np.random.rand(n_samples, *image_shape)
    
    # save random data as png
    for i in range(n_samples):
        img = Image.fromarray( (batch_x[i]*255).astype('uint8') )
        img.save(f'{data_path}/{i}.png')


def load_preprocessed_data(
        data_path, 
        movement_mode,
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        multiplier,
        n_rotations,
        preprocess_func=None,
        color_mode='rgb', 
        target_size=(224, 224), 
        image_shape=(224, 224, 3),
        interpolation='nearest',
        data_format='channels_last',
        dtype=K.floatx(),
    ):
    if movement_mode == '1d':
        n_stops_x = len( range(env_x_min*multiplier, env_x_max*multiplier+1) )
        n_samples = n_stops_x

    elif movement_mode == '2d':
        n_stops_x = len( range(env_x_min*multiplier, env_x_max*multiplier+1) )
        n_stops_y = len( range(env_y_min*multiplier, env_y_max*multiplier+1) )
        n_samples = (n_stops_x * n_stops_y) * n_rotations

    print(f'[Check] data loader - n_samples: {n_samples}')

    # build batch of image data
    batch_x = np.zeros((n_samples,) + image_shape, dtype=dtype)

    # TODO: careful for vit, might have image_shape=(3, 224, 224)

    for i in range(n_samples):
        fpath = f'{data_path}/{i}.png'
        # PIL.Image
        img = load_img(
            fpath,
            color_mode=color_mode,
            target_size=target_size,
            interpolation=interpolation)
        x = img_to_array(img, data_format=data_format)

        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, 'close'):
            img.close()
        if preprocess_func:

            # TODO: for vit, might need to do inputs['pixel_values']

            x = preprocess_func(x)
        batch_x[i] = x
    
    print(f'[Check] data loader - batch_x.shape: {batch_x.shape}')
    return batch_x


def load_decoding_targets(
        movement_mode,
        env_x_min,
        env_x_max,
        env_y_min,
        env_y_max,
        multiplier,
        n_rotations
    ):
    """
    Produce coordinates as targets for training/testing

    return:
        A list of lists, where each sublist is a coordinate
        corresponding to a frame and a rotation.
    """

    targets_true = []
    if movement_mode == '1d':
        NotImplementedError()

    elif movement_mode == '2d':
        # same idea as generating the frames in Unity
        # so we get decimal coords in between the grid points
        for i in range(env_x_min*multiplier, env_x_max*multiplier+1):
            for j in range(env_y_min*multiplier, env_y_max*multiplier+1):
                for k in range(n_rotations):
                    targets_true.append([i/multiplier, j/multiplier, k])
    
    return np.array(targets_true)


def load_full_dataset_model_reps(
        config, model, preprocessed_data
    ):
    """
    Generic function to produce model representations
    for the full dataset (before train/test split)
    """    
    # use raw image input 
    if config['model_name'] == 'none':
        model_reps = preprocessed_data.reshape(preprocessed_data.shape[0], -1)
        print(f'raw image input shape: {model_reps.shape}')
    # use model output
    else:
        # (n, 4096)

        # TODO: for vit, config['output_layer'] needs to be 
        # accessed here in order to get the correct output.

        # TODO: think about how reps shape diff in VIT 
        # and whether decoding works out of box from flattened hidden layer outputs.

        model_reps = model.predict(preprocessed_data, verbose=1)

        # NOTE: solution to OOM for early layers is to save batches to disk
        # and merge on CPU and do whatever operations come below.
        del model
        K.clear_session()
        if len(model_reps.shape) > 2:
            # when not a fc layer, we need to flatten the output dim
            # except the batch dim.
            model_reps = model_reps.reshape(model_reps.shape[0], -1)
        print(f'model_reps.shape: {model_reps.shape}')
    return model_reps


def load_model_layers(model_name):
    """
    Given `model_name`, return the main
    layers of the model that are of interest
    for analysis.
    """
    models_and_layers = {
        'simclrv2_r50_1x_sk0':
            [
                'final_avg_pool',
                'block_group4',
                'block_group2',
                'block_group1'
            ],
        'vgg16':
            [   
                'fc2',
                'block5_pool',
                'block4_pool',
                # 'block2_pool',
            ],
        'resnet50':
            [
                'avg_pool',
                'conv5_block2_out',
                'conv4_block6_out',
                'conv2_block3_out',
            ],
        # TODO: vit_b16 requires layer naming.
    }
    return models_and_layers[model_name]
    

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
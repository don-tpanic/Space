import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array


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


def load_data(
        data_path, 
        movement_mode,
        x_min,
        x_max,
        y_min,
        y_max,
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
        n_stops_x = len( range(x_min*multiplier, x_max*multiplier+1) )
        n_samples = n_stops_x

    elif movement_mode == '2d':
        n_stops_x = len( range(x_min*multiplier, x_max*multiplier+1) )
        n_stops_y = len( range(y_min*multiplier, y_max*multiplier+1) )
        n_samples = (n_stops_x * n_stops_y) * n_rotations

    print(f'[Check] data loader - n_samples: {n_samples}')

    # build batch of image data
    batch_x = np.zeros((n_samples,) + image_shape, dtype=dtype)
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
            x = preprocess_func(x)
        batch_x[i] = x
    
    print(f'[Check] data loader - batch_x.shape: {batch_x.shape}')
    return batch_x


def load_data_targets(
        movement_mode,
        x_min,
        x_max,
        y_min,
        y_max,
        multiplier,
        n_rotations
    ):
    """
    Produce coordinates as targets for training/testing

    return:
        A list of lists, where each sublist is a coordinate
        corresponding to a frame and a rotation.
    """

    coords_true = []
    if movement_mode == '1d':
        NotImplementedError()

    elif movement_mode == '2d':
        # same idea as generating the frames in Unity
        # so we get decimal coords in between the grid points
        for i in range(x_min*multiplier, x_max*multiplier+1):
            for j in range(y_min*multiplier, y_max*multiplier+1):
                for k in range(n_rotations):
                    coords_true.append([i/multiplier, j/multiplier, k])
    
    return np.array(coords_true)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import utils
    config_version = "env8_2d_none_raw_9_pca"
    config = utils.load_config(config_version)
    unity_env = config['unity_env']
    model_name = config['model_name']
    movement_mode = config['movement_mode']
    n_rotations = config['n_rotations']
    x_min = config['x_min']
    x_max = config['x_max']
    y_min = config['y_min']
    y_max = config['y_max']
    multiplier = config['multiplier']
    data_path = f"data/unity/{unity_env}/{movement_mode}"

    # generate_random_data(
    #     data_path=data_path,
    #     movement_mode=movement_mode,
    #     x_min=x_min,
    #     x_max=x_max,
    #     y_min=y_min,
    #     y_max=y_max,
    #     multiplier=multiplier,
    #     n_rotations=n_rotations,
    #     random_seed=random_seed,
    # )

    # load_data(
    #     data_path, 
    #     movement_mode,
    #     x_min,
    #     x_max,
    #     y_min,
    #     y_max,
    #     multiplier,
    #     n_rotations,
    #     preprocess_func=None
    # )

    load_data_targets(
        movement_mode,
        x_min,
        x_max,
        y_min,
        y_max,
        multiplier,
        n_rotations
    )
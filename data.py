import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_data(
        data_path,
        movement_mode,
        preprocess_func=None,
        color_mode='rgb', 
        target_size=(224, 224), 
        image_shape=(224, 224, 3),
        interpolation='nearest',
        data_format='channels_last',
        dtype=K.floatx(),
    ):
    ###########
    # Unity env dimension (Unity z axis in python is now y axis)
    x_min = -4
    x_max = 4
    y_min = -4
    y_max = 4
    ###########

    if movement_mode == '1d':
        multiplier = 10
        n_stops_x = len( range(x_min*multiplier, x_max*multiplier+1) )
        n_samples = n_stops_x

    elif movement_mode == '2d':
        multiplier = 2
        n_rotations = 6
        n_stops_x = len( range(x_min*multiplier, x_max*multiplier+1) )
        n_stops_y = len( range(y_min*multiplier, y_max*multiplier+1) )
        n_samples = (n_stops_x * n_stops_y) * n_rotations

    print(f'n_samples: {n_samples}')

    # build batch of image data
    batch_x = np.zeros((n_samples,) + image_shape, dtype=dtype)
    for i in range(n_samples):
        fpath = f'{data_path}/{movement_mode}/{i}.png'
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
    
    print(f'batch_x.shape: {batch_x.shape}')
    return batch_x


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    movement_mode = '1d'
    data_path = f'data/unity'
    load_data(data_path, movement_mode)
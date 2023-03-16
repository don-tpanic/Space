import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import tensorflow as tf
from tensorflow.keras.models import Model

"""
Model loading function and definitions.
"""

def load_model(model_name, output_layer):
    if model_name == 'vgg16':
        model = tf.keras.applications.VGG16(
            weights='imagenet', 
            include_top=True, 
            input_shape=(224, 224, 3))
        preprocess_func = tf.keras.applications.vgg16.preprocess_input
    
    model = Model(inputs=model.input, outputs=model.get_layer(output_layer).output)
    # model.summary()
    return model, preprocess_func


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = load_model()
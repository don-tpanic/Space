import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import tensorflow as tf
from tensorflow.keras.models import Model

"""
Model loading function and definitions.
"""

def load_model(model_name, output_layer=None, input_shape=(224, 224, 3)):
    if 'simclr' in model_name:
        if model_name == 'simclrv2_r50_1x_sk0':
            model_path = f'model_zoo/{model_name}/saved_model'
            model = _build_simclr(model_path, output_layer)
            from model_zoo.simclrv2_r50_1x_sk0 import preprocessing
            preprocess_func = preprocessing.preprocess_func
    else:
        if 'vit' in model_name:
            if model_name == 'vit_b16':
                from transformers import AutoImageProcessor, TFViTModel
                model = TFViTModel.from_pretrained(
                    'google/vit-base-patch16-224-in21k',
                    cache_dir='model_zoo/vit_b16'
                )
                preprocess_func = AutoImageProcessor.from_pretrained(
                    "google/vit-base-patch16-224-in21k",
                    cache_dir='model_zoo/vit_b16'
                )
            elif model_name == 'vit_b16_untrained':
                from transformers import AutoImageProcessor, ViTConfig, TFViTModel                        
                config = ViTConfig()
                model = TFViTModel(config)
                preprocess_func = AutoImageProcessor.from_pretrained(
                    "google/vit-base-patch16-224-in21k",
                    cache_dir='model_zoo/vit_b16'
                )

        else:
            if model_name == 'vgg16':
                model = tf.keras.applications.VGG16(
                    weights='imagenet', 
                    include_top=True, 
                    input_shape=input_shape,
                    classifier_activation=None
                )
                preprocess_func = tf.keras.applications.vgg16.preprocess_input

            elif model_name == 'vgg16_untrained':
                model = tf.keras.applications.VGG16(
                    weights=None, 
                    include_top=True, 
                    input_shape=input_shape,
                    classifier_activation=None
                )
                preprocess_func = tf.keras.applications.vgg16.preprocess_input

            elif model_name == 'resnet50':
                model = tf.keras.applications.ResNet50(
                    weights='imagenet', 
                    include_top=True, 
                    input_shape=input_shape,
                    classifier_activation=None
                )
                preprocess_func = tf.keras.applications.resnet50.preprocess_input

            elif model_name == 'resnet50_untrained':
                model = tf.keras.applications.ResNet50(
                    weights=None, 
                    include_top=True, 
                    input_shape=input_shape,
                    classifier_activation=None
                )
                preprocess_func = tf.keras.applications.resnet50.preprocess_input

            if output_layer is None:
                output_layer = model.layers[-1].name
            model = Model(inputs=model.input, outputs=model.get_layer(output_layer).output)

    return model, preprocess_func


def _build_simclr(model_path, output_layer):
    class SimCLRv2(tf.keras.Model):
        def __init__(self):
            super(SimCLRv2, self).__init__()
            self.saved_model = \
                tf.saved_model.load(model_path)
            self.output_layer = output_layer

        def call(self, inputs):
            # print((self.saved_model(
            #         inputs, trainable=False)).keys())
            return \
                self.saved_model(
                    inputs, trainable=False)[self.output_layer]
    return SimCLRv2()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model, preprocess_func = load_model(model_name='vit_b16')

    print(
    # model.vit._layers[1]                                              # is TFViTEncoder, 
    # model.vit._layers[1]._layers                                      # contain 12 TFViTLayer
    # model.vit._layers[1]._layers[0][0])                               # 1/12 TFViTLayer (Transformer block)
    # model.vit._layers[1]._layers[0][0]._layers,                       # has Attention, layernorm, etc.
    # model.vit._layers[1]._layers[0][0]._layers[0]._layers,            # TFViTSelfAttention+FViTSelfOutput
    model.vit._layers[1]._layers[0][0]._layers[0]._layers[0]._layers,   # 3 Dense layers and dropout (Q,K,V)
    )

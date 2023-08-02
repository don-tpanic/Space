import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import tensorflow as tf
from tensorflow.keras.models import Model

"""
Model loading function and definitions.
"""

def load_model(model_name, output_layer=None, input_shape=(224, 224, 3)):
    if 'vit' in model_name:
        if model_name == 'vit_b16':
            from transformers import AutoImageProcessor, TFViTModel
            model = TFViTModel.from_pretrained(
                'google/vit-base-patch16-224-in21k',
                cache_dir='/home/ken/projects/Space/model_zoo/vit_b16'
            )
            preprocess_func = AutoImageProcessor.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                cache_dir='/home/ken/projects/Space/model_zoo/vit_b16'
            )
        elif model_name == 'vit_b16_untrained':
            from transformers import AutoImageProcessor, ViTConfig, TFViTModel                        
            config = ViTConfig()
            model = TFViTModel(config)
            preprocess_func = AutoImageProcessor.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                cache_dir='model_zoo/vit_b16'
            )

    return model, preprocess_func


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # vit shapes
    model, preprocess_func = load_model(model_name='vit_b16')
    synthetic_data = tf.random.uniform((1, 3, 224, 224))
    outs = model(
                synthetic_data, training=False, 
                output_hidden_states=True
            )

    print(
        outs.keys(), '\n\n',  
        # ['last_hidden_state', 'pooler_output', 'hidden_states']
        model.layers, '\n\n',  
        # [<transformers.models.vit.modeling_tf_vit.TFViTMainLayer object at 0x7f4b16a53bb0>]
        model.layers[0]._layers, '\n\n',  
        # [<transformers.models.vit.modeling_tf_vit.TFViTEmbeddings object at 0x7f414e06bdf0>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTEncoder object at 0x7f414c1288b0>, 
        # <tensorflow.python.keras.layers.normalization.LayerNormalization object at 0x7f41402ccfd0>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTPooler object at 0x7f41402f02e0>, 
        # DictWrapper({'add_pooling_layer': True, 'name': 'vit'})]
        model.layers[0]._layers[0]._layers, '\n\n',
        # [<transformers.models.vit.modeling_tf_vit.TFViTPatchEmbeddings object at 0x7ff2d41670a0>, 
        # <tensorflow.python.keras.layers.core.Dropout object at 0x7ff2d4167880>] 
        model.layers[0]._layers[1]._layers, '\n\n',
        # [ListWrapper([<transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d4167c10>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d41784c0>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d40b4d00>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d40cb580>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d40d7dc0>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d406e640>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d407be80>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d408f700>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d401ef40>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d40327f0>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d4041f70>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTLayer object at 0x7ff2d40568b0>])]
        model.layers[0]._layers[1]._layers[0][0]._layers, '\n\n',
        # [<transformers.models.vit.modeling_tf_vit.TFViTAttention object at 0x7f272010ee50>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTIntermediate object at 0x7f27200ae4f0>, 
        # <transformers.models.vit.modeling_tf_vit.TFViTOutput object at 0x7f27200ae9d0>,
        # <tensorflow.python.keras.layers.normalization.LayerNormalization object at 0x7f27200aeeb0>, 
        # <tensorflow.python.keras.layers.normalization.LayerNormalization object at 0x7f272012e340>] 
    )

    # weights 
    print(
        model.layers[0]._layers[1]._layers[0][0]._layers[0].name, '\n\n',  # attention
        model.layers[0]._layers[1]._layers[0][0]._layers[0].weights, '\n\n',
    )

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import utils
import models
import data


def _single_model_reps(config):
    """
    Produce model_reps either directly computing if the first time,
    or load from disk if already computed.

    return:
        model_reps: \in (n_locations, n_rotations, n_features)
    """
    model_reps_fname = \
        f'results/'\
        f'{config["unity_env"]}/'\
        f'{config["movement_mode"]}/'\
        f'{config["model_name"]}/'\
        f'model_reps_{config["output_layer"]}.npy'

    if os.path.exists(model_reps_fname):
        logging.info(f'Loading model_reps from {model_reps_fname}')
        model_reps = np.load(model_reps_fname)
        logging.info(f'model_reps.shape: {model_reps.shape}')
        return model_reps

    else:
        # load model outputs
        if config['model_name'] == 'none':
            model = None
            preprocess_func = None
        else:
            model, preprocess_func = models.load_model(
                config['model_name'], config['output_layer'])

        preprocessed_data = data.load_preprocessed_data(
            config=config,
            data_path=\
                f"data/unity/"\
                f"{config['unity_env']}/"\
                f"{config['movement_mode']}", 
            movement_mode=config['movement_mode'],
            env_x_min=config['env_x_min'],
            env_x_max=config['env_x_max'],
            env_y_min=config['env_y_min'],
            env_y_max=config['env_y_max'],
            multiplier=config['multiplier'],
            n_rotations=config['n_rotations'],
            preprocess_func=preprocess_func,
        )    
        
        # (n_locations*n_rotations, n_features)
        model_reps = data.load_full_dataset_model_reps(
            config, model, preprocessed_data
        )

        # reshape to (n_locations, n_rotations, n_features)
        model_reps = model_reps.reshape(
            (model_reps.shape[0] // config['n_rotations'],  # n_locations
            config['n_rotations'],                          # n_rotations
            model_reps.shape[1])                            # all units
        )

        # save to disk  
        # # TODO: slow to save but is it benefitcial as we do it only once?
        # logging.info(f'Saving model_reps to {model_reps_fname}...')
        # np.save(model_reps_fname, model_reps)
        # logging.info(f'[Saved] model_reps to {model_reps_fname}')
        return model_reps


config = utils.load_config('env28_r24_2d_vgg16_fc2')
model_reps = _single_model_reps(config)
model_reps_summed = np.sum(model_reps, axis=1, keepdims=True)

# sum all rows 
model_reps_summed_summed = np.sum(model_reps_summed, axis=0)
print(model_reps_summed_summed.shape)

# print how many zeros
print(np.sum(model_reps_summed_summed == 0) / model_reps_summed_summed.shape[0])
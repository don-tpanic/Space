import yaml
from utils import load_config


def generate_config(
        template_config,
        env_nums,
        n_rotations_list,
        movement_modes,
        model_names,
        output_layers,
        feature_selections,
    ):
    template = load_config(template_config)
    for env_num in env_nums:
        for n_rotations in n_rotations_list:
            for movement_mode in movement_modes:
                for model_name in model_names:
                    for output_layer in output_layers:
                        for feature_selection in feature_selections:
                            config_version = \
                                f'env{env_num}_r{n_rotations}_{movement_mode}_' \
                                f'{model_name}_{output_layer}_' \
                                f'{feature_selection}'
                            
                            template['config_version'] = config_version
                            template['unity_env'] = f'env{env_num}_r{n_rotations}'
                            template['n_rotations'] = n_rotations
                            template['movement_mode'] = movement_mode
                            template['model_name'] = model_name
                            template['output_layer'] = output_layer
                            template['feature_selection'] = feature_selection
                            with open(f'configs/{config_version}.yaml', 'w') as f:
                                yaml.dump(template, f, sort_keys=False)


if __name__ == "__main__":
    generate_config(
        template_config='env28_r24_2d_vgg16_fc2_full',
        env_nums=['28'],
        n_rotations_list=[24],
        movement_modes=['2d'],
        model_names=['simclrv2_r50_1x_sk0'],
        output_layers=[
            'final_avg_pool', 
            'block_group4', 
            'block_group2', 
            'block_group1'
        ],
        feature_selections=['full'],
    )
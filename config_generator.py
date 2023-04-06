import yaml
from utils import load_config


def generate_config(
        template_config,
        env_nums,
        n_rotations_list=[24],
        movement_modes=['2d'],
        model_names=['none'],
        output_layers=['raw'],
        reduction_methods=['maxvar', 'pca', 'nmf'],
        n_components=9,
    ):
    template = load_config(template_config)
    for env_num in env_nums:
        for n_rotations in n_rotations_list:
            for movement_mode in movement_modes:
                for model_name in model_names:
                    for output_layer in output_layers:
                        for reduction_method in reduction_methods:
                            config_version = \
                                f'env{env_num}_r{n_rotations}_{movement_mode}_' \
                                f'{model_name}_{output_layer}_{n_components}_' \
                                f'{reduction_method}'
                            
                            template['config_version'] = config_version
                            template['unity_env'] = f'env{env_num}_r{n_rotations}'
                            template['n_rotations'] = n_rotations
                            template['movement_mode'] = movement_mode
                            template['model_name'] = model_name
                            template['output_layer'] = output_layer
                            template['reduction_method'] = reduction_method
                            template['n_components'] = n_components
                            with open(f'configs/{config_version}.yaml', 'w') as f:
                                yaml.dump(template, f, sort_keys=False)


if __name__ == "__main__":
    generate_config(
        template_config='env28_r24_2d_vgg16_fc2_50_pca',
        env_nums=['28', '33'],
        model_names=['vgg16'],
        output_layers=['fc2'],
        reduction_methods=['minvar'],
        n_components=50,
    )
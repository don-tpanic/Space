import os
import multiprocessing
from heatmap_analysis import execute
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    num_processes = 70
    envs = [3, 8, 9]
    movement_modes = ['2d']
    dim_reductions = ['pca', 'nmf']
    n_components_list = [9]
    model_types_n_reps = {'vgg16': 'fc2', 'none': 'raw'}

    with multiprocessing.Pool(num_processes) as pool:
        for env in envs:
            for movement_mode in movement_modes:
                for dim_reduction in dim_reductions:
                    for n_components in n_components_list:
                        for model_type, model_rep in model_types_n_reps.items():
                            config_version = \
                                f'env{env}_{movement_mode}_{model_type}_{model_rep}_' \
                                f'{n_components}_{dim_reduction}'

                            results = pool.apply_async(
                                execute, 
                                args=[config_version]
                            )
                
        print(results.get())
        pool.close()
        pool.join()
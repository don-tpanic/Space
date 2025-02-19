import os
import yaml
import multiprocessing


def load_config(config_version):
    with open(os.path.join(f'configs', f'{config_version}.yaml')) as f:
        config = yaml.safe_load(f)
    return config


def load_results_path(
        config, 
        experiment, 
        feature_selection=None, 
        decoding_model_choice=None,
        sampling_rate=None,
        moving_trajectory=None,
        random_seed=None,
        reference_experiment=None,
    ):
    """
    Notes:
        reference_experiment:
            only need to be provided if we are inspecting units
            as inspecting units depends on whether the units are
            selected from the loc_n_rot experiment or the border
            experiment.

        if experiment == 'unit_chart':
            feature_selection, decoding_model_choice, sampling_rate, random_seed,
            and reference_experiment are not needed. This is because unit_chart
            is of the level of model regardless of experiments and is only 
            specific to model layers.
    """
    unity_env = config['unity_env']
    model_name = config['model_name']
    output_layer = config['output_layer']
    movement_mode = config['movement_mode']

    if experiment in ['loc_n_rot', 'border_dist']:
        decoding_model_name = decoding_model_choice['name']
        decoding_model_hparams = decoding_model_choice['hparams']
        results_path = \
            f'results/{unity_env}/{movement_mode}/{moving_trajectory}/'\
            f'{model_name}/{experiment}/{feature_selection}/'\
            f'{decoding_model_name}_{decoding_model_hparams}/'\
            f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
        
    elif experiment in ['unit_chart_by_coef'] \
            and reference_experiment in ['loc_n_rot', 'border_dist']:
        decoding_model_name = decoding_model_choice['name']
        decoding_model_hparams = decoding_model_choice['hparams']
        results_path = \
            f'results/{unity_env}/{movement_mode}/{moving_trajectory}/'\
            f'{model_name}/inspect_units/{reference_experiment}/{experiment}/{feature_selection}/'\
            f'{decoding_model_name}_{decoding_model_hparams}/'\
            f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
    
    elif experiment == 'unit_chart':
        # no reference_experiment needed as its general to all experiments
        # but still belongs to inspect_units
        results_path = \
            f'results/{unity_env}/{movement_mode}/{moving_trajectory}/'\
            f'{model_name}/inspect_units/{experiment}/{output_layer}'
        
    if not os.path.exists(results_path):
        if experiment == 'unit_chart':
            os.makedirs(results_path)
        else:
            if \
            (
                'l1' in feature_selection and \
                decoding_model_choice['name'] != 'lasso_regression'
            ) \
            or \
            (
                'l2' in feature_selection and \
                decoding_model_choice['name'] != 'ridge_regression'
            ):  
                # do not create dir if mismatch between 
                # feature selection and decoding model
                results_path = None
            else:
                os.makedirs(results_path)
    return results_path


def load_figs_path(
        config, 
        experiment, 
        feature_selection=None, 
        decoding_model_choice=None,
        sampling_rate=None,
        moving_trajectory=None,
        random_seed=None,
        reference_experiment=None,
    ):
    """
    Notes:
        reference_experiment:
            only need to be provided if we are inspecting units
            as inspecting units depends on whether the units are
            selected from the loc_n_rot experiment or the border
            experiment.

        if experiment == 'unit_chart':
            feature_selection, decoding_model_choice, sampling_rate, random_seed,
            and reference_experiment are not needed. This is because unit_chart
            is of the level of model regardless of experiments and is only 
            specific to model layers.
    """
    unity_env = config['unity_env']
    model_name = config['model_name']
    output_layer = config['output_layer']
    movement_mode = config['movement_mode']

    if experiment in ['loc_n_rot', 'border_dist']:
        decoding_model_name = decoding_model_choice['name']
        decoding_model_hparams = decoding_model_choice['hparams']
        figs_path = \
            f'figs/{unity_env}/{movement_mode}/{moving_trajectory}/'\
            f'{model_name}/{experiment}/{feature_selection}/'\
            f'{decoding_model_name}_{decoding_model_hparams}/'\
            f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
        
    elif experiment in ['unit_chart_by_coef'] \
            and reference_experiment in ['loc_n_rot', 'border_dist']:
        decoding_model_name = decoding_model_choice['name']
        decoding_model_hparams = decoding_model_choice['hparams']
        figs_path = \
            f'figs/{unity_env}/{movement_mode}/{moving_trajectory}/'\
            f'{model_name}/inspect_units/{reference_experiment}/{experiment}/{feature_selection}/'\
            f'{decoding_model_name}_{decoding_model_hparams}/'\
            f'{output_layer}/sr{sampling_rate}/seed{random_seed}'
    
    elif experiment == 'unit_chart':
        # no reference_experiment needed as its general to all experiments
        # but still belongs to inspect_units
        figs_path = \
            f'figs/{unity_env}/{movement_mode}/{moving_trajectory}/'\
            f'{model_name}/inspect_units/{experiment}/{output_layer}'
        
    if not os.path.exists(figs_path):
        if experiment == 'unit_chart':
            os.makedirs(figs_path)
        else:
            if \
            (
                'l1' in feature_selection and \
                decoding_model_choice['name'] != 'lasso_regression'
            ) \
            or \
            (
                'l2' in feature_selection and \
                decoding_model_choice['name'] != 'ridge_regression'
            ):  
                # do not create dir if mismatch between 
                # feature selection and decoding model
                figs_path = None
            else:
                os.makedirs(figs_path)
    return figs_path


def cuda_manager(target, args_list, cuda_id_list, n_concurrent=None):
    """Create CUDA manager.
    Arguments:
        target: A target function to be evaluated.
        args_list: A list of dictionaries, where each dictionary
            contains the arguments necessary for the target function.
        cuda_id_list: A list of eligable CUDA IDs.
        n_concurrent (optional): The number of concurrent CUDA
            processes allowed. By default this is equal to the length
            of `cuda_id_list`.
    Raises:
        Exception
    """
    if n_concurrent is None:
        n_concurrent = len(cuda_id_list)
    else:
        n_concurrent = min([n_concurrent, len(cuda_id_list)])

    shared_exception = multiprocessing.Queue()

    n_task = len(args_list)

    args_queue = multiprocessing.Queue()
    for args in args_list:
        args_queue.put(args)

    # Use a semaphore to make one child process per CUDA ID.
    # NOTE: Using a pool of workers may not work with TF because it
    # re-uses existing processes, which may not release the GPU's memory.
    sema = multiprocessing.BoundedSemaphore(n_concurrent)

    # Use manager to share list of available CUDA IDs among child processes.
    with multiprocessing.Manager() as manager:
        available_cuda = manager.list(cuda_id_list)

        process_list = []
        for _ in range(n_task):
            process_list.append(
                multiprocessing.Process(
                    target=cuda_child,
                    args=(
                        target, args_queue, available_cuda, shared_exception,
                        sema
                    )
                )
            )

        for p in process_list:
            p.start()

        for p in process_list:
            p.join()

    #  Check for raised exceptions.
    e_list = [shared_exception.get() for _ in process_list]
    for e in e_list:
        if e is not None:
            raise e


def cuda_child(target, args_queue, available_cuda, shared_exception, sema):
    """Create child process of the CUDA manager.
    Arguments:
        target: The function to evaluate.
        args_queue: A multiprocessing.Queue that yields a dictionary
            for consumption by `target`.
        available_cuda: A multiprocessing.Manager.list object for
            tracking CUDA device availablility.
        shared_exception: A multiprocessing.Queue for exception
            handling.
        sema: A multiprocessing.BoundedSemaphore object ensuring there
            are never more processes than eligable CUDA devices.
    """
    try:
        sema.acquire()
        args = args_queue.get()
        cuda_id = available_cuda.pop()

        os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(cuda_id)

        target(**args)

        shared_exception.put(None)
        available_cuda.append(cuda_id)
        sema.release()

    except Exception as e:
        shared_exception.put(e)

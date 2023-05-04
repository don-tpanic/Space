import numpy as np
import matplotlib.pyplot as plt
import utils


def investigate_1(
        config_version,
        sampling_rates
    ):
    """
    After a certain sampling rate, decoding error for location
    explodes. A check on coef magnitudes w 
        `single_env_regression_weights_across_sampling_rates`
    reveals that are coef are huge in magnitudes. 
    This function investigates this issue by looking into the 
    features with large coef. 
    
    Impl:
        For a given env and sampling rate,
        1. Load X_train from disk.
        2. Load regression coef from disk.
        3. Plot the entire X_train as heatmap.
    """
    results_path = utils.load_results_path(config_version)
    for sampling_rate in sampling_rates:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)

        # load X_train
        X_train = np.load(
            f'{results_path}/X_train_{sampling_rate}.npy')
        
        # load regression coef
        coef = np.load(
            f'{results_path}/coef_{sampling_rate}.npy')
        
        # coef \in (n_targets, n_features) where 0th target is x,
        # to investigate, we use 0th column and we find the top 500 highest
        # coef features' indices and we extract the corresponding columns
        # from X_train and plot the heatmap.
        coef_x = coef[0, :]
        top_500_coef_x_indices = np.argsort(coef_x)[::-1][:500]
        top_500_X_train = X_train[:, top_500_coef_x_indices]
        heatmap = ax.imshow(top_500_X_train, cmap='viridis')
        ax.set_title(f'X_train heatmap for sampling rate {sampling_rate}')
        ax.set_xlabel('units')
        ax.set_ylabel('samples')
        ax.set_xlim(-0.5, 500.5)
        plt.colorbar(heatmap)
        plt.savefig(f'{results_path}/X_train_heatmap_{sampling_rate}.png')

        # plot the first 6 columns of X_train from above
        # as histogram, each column is a subplot of 2*3
        fig, ax = plt.subplots(2, 3, figsize=(10, 10), dpi=100)
        for i in range(6):
            ax[i//3, i%3].hist(top_500_X_train[:, i], bins=100)
            ax[i//3, i%3].set_xlabel('units activation')

            # compute proportion of units that are 0,
            # and plot as title
            prop_zero = np.sum(top_500_X_train[:, i]==0) / \
                top_500_X_train.shape[0]
            ax[i//3, i%3].set_title(
                f'coef_x={coef_x[top_500_coef_x_indices[i]]:.1f},' \
                f'prop_zero={prop_zero:.4f}'
            )
        plt.savefig(f'{results_path}/X_train_hist_{sampling_rate}.png')


def investigate_2(config_version):
    """
    Based on investigate_1, we found that the large
    coef correspond to units that are mostly 0 across
    samples. In fact, as sampling rate increases, 
    the proportion of 0s increases, which is somewhat puzzling.

    This function investigates the entire `model_reps` which is 
    sampling_rate agnostic. We will then manually sample rows 
    given different sampling rates (using different seeds) to see
    if sampling rate increases will lead to proportion of 0s increases.
    """
    results_path = utils.load_results_path(config_version)
    model_reps = np.load(f'{results_path}/model_reps.npy')
    print(
        f'number of columns of model_reps that are all 0s: ' \
        f'{np.sum(np.all(model_reps==0, axis=0))}' \
        f'/{model_reps.shape[1]}'
    )

    # Given different seeds and sampling rates, we sample rows of model_reps
    # and we compute number of columns are all 0s. For each seed, we plot
    # the number of columns that are all 0s against sampling rates where each
    # seed is a subplot.
    seeds = [0, 1234, 666, 42, 999]
    sampling_rates = np.linspace(0.1, 1.0, 10)
    fig, ax = plt.subplots(5, 1, figsize=(10, 10), dpi=100)
    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        for sampling_rate in sampling_rates:
            sampled_rows = np.random.choice(
                model_reps.shape[0], 
                size=int(model_reps.shape[0]*sampling_rate),
                replace=False
            )
            sampled_model_reps = model_reps[sampled_rows, :]
            num_all_zero_cols = np.sum(np.all(sampled_model_reps==0, axis=0))
            ax[i].scatter(sampling_rate, num_all_zero_cols, color='k')
            ax[i].set_xlabel('sampling rate')
            ax[i].set_ylabel('number of all 0s columns')
            ax[i].set_title(f'seed={seed}')
    plt.savefig(f'{results_path}/model_reps_all_0s.png')


def investigate_3(config_version):
    """
    Investigate_2 reminded us that in fact if we randomly sample rows,
    the lower the sampling rate, the more cols will be all 0s. 
    Why? Because the chances of getting all 0s in a shorter column (i.e. small sampling rate)
    is a lot higher than getting all 0s in a longer column. So how come we see the opposite
    pattern in the original experiment? The difference to purely randomly sample raws,
    in the original experiment we sample locations first and add all rotations of the same 
    location, would that make a difference? This function investigates that.
    """
    results_path = utils.load_results_path(config_version)
    model_reps = np.load(f'{results_path}/model_reps.npy')
    print(
        f'number of columns of model_reps that are all 0s: ' \
        f'{np.sum(np.all(model_reps==0, axis=0))}' \
        f'/{model_reps.shape[1]}'
    )

    # Given different seeds and sampling rates, we sample rows of model_reps
    # and we compute number of columns are all 0s. For each seed, we plot
    # the number of columns that are all 0s against sampling rates where each
    # seed is a subplot. The difference to investigate 2 is we sample locations
    # first and add all rotations of the same location.
    seeds = [0, 1234, 666, 42, 999]
    sampling_rates = np.linspace(0.1, 1.0, 10)
    n_rotations = 24
    
    fig, ax = plt.subplots(5, 1, figsize=(10, 10), dpi=100)
    for seed_index, seed in enumerate(seeds):
        np.random.seed(seed)
        for sampling_rate in sampling_rates:

            sampled_locs = np.random.choice(
                model_reps.shape[0] // n_rotations, 
                size=int(model_reps.shape[0]*sampling_rate//n_rotations),
                replace=False
            )

            sampled_locs_all_rots = []
            for i in sampled_locs:
                sampled_locs_all_rots.extend(
                    [i*n_rotations + j for j in range(n_rotations)]
                )
            
            sampled_model_reps = model_reps[sampled_locs_all_rots, :]
            num_all_zero_cols = np.sum(np.all(sampled_model_reps==0, axis=0))

            ax[seed_index].scatter(sampling_rate, num_all_zero_cols, color='k')
            ax[seed_index].set_xlabel('sampling rate')
            ax[seed_index].set_ylabel('number of all 0s columns')
            ax[seed_index].set_title(f'seed={seed}')
    plt.savefig(f'{results_path}/model_reps_all_0s_2.png')
    

if __name__ == '__main__':
    investigate_1(
        config_version='env28_r24_2d_vgg16_fc2',
        sampling_rates=[0.6]
    )
    ### confirmed: the large coef correspond to units that are mostly 0 across

    investigate_2(config_version='env28_r24_2d_vgg16_fc2')
    ### confirmed: sampling rows randomly will lead to less cols that are all 0s

    investigate_3(config_version='env28_r24_2d_vgg16_fc2')   
    ### confirmed: sampling loc and then add rotations has no effect different 
    # from randomly sampling rows in 2.
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import utils


def plot_components(stats, config_version):
    """
    Plot individual components of each rotation separately,
    up to the specified `n_components` and `n_rotations`.

    Impl:

    Return:

    """
    config = utils.load_config(config_version)
    n_components=config['n_components']
    n_rotations=config['n_rotations']
    movement_mode=config['movement_mode']
    model_name=config['model_name']
    output_layer=config['output_layer']
    reduction_method=config['reduction_method']
    x_min=config['x_min']
    x_max=config['x_max']
    y_min=config['y_min']
    y_max=config['y_max']
    multiplier=config['multiplier']
    results_path = utils.return_results_path(config_version)

    # prepare coordinates for plotting heatmap
    if movement_mode == '1d':
        x_axis_coords = []
        y_axis_coords = []
        for i in range(x_min*multiplier, x_max*multiplier+1):
            x_axis_coords.append(i/multiplier)
            y_axis_coords.append(0)

    elif movement_mode == '2d':
        x_axis_coords = []
        y_axis_coords = []
        # same idea as generating the frames in Unity
        # so we get decimal coords in between the grid points
        for i in range(x_min*multiplier, x_max*multiplier+1):
            for j in range(y_min*multiplier, y_max*multiplier+1):
                x_axis_coords.append(i/multiplier)
                y_axis_coords.append(j/multiplier)

    # the fig has subplots 
    # of the components for each rotation
    fig, ax = plt.subplots(
        n_components, 
        n_rotations, 
        figsize=(int(3*n_rotations), int(3*n_components))
    )

    if n_rotations > 1:

        for i in range(n_components):
            # each is a matrix of (n_locations, n_rotations)
            component = np.load(f'{results_path}/components_{i+1}.npy')

            if reduction_method == 'pca':
                per_component_variance_explained = stats[i]

            for j in range(n_rotations):  # i.e. n_rotations
                component_per_rotation = component[:, j]

                ax[i, j].scatter(
                    x_axis_coords, y_axis_coords, 
                    c=component_per_rotation, cmap='viridis', s=144)
                ax[i, j].set_title(f'c{i+1}r{j+1}')
                ax[i, j].set_xlim(x_min, x_max)
                ax[i, j].set_ylim(y_min, y_max)

                # # turn off unnec axes
                if i < n_components-1:
                    ax[i, j].set_xticks([])
                if j > 0:
                    ax[i, j].set_yticks([])
            
            if reduction_method == 'pca':
                # plot variance explained per component on the right most column
                ax2 = ax[i, -1].twinx()
                ax2.set_ylabel(f'{per_component_variance_explained:.4f}', rotation=180)

        ax[n_components//2, 0].set_ylabel('Unity z axis')
        ax[-1, n_rotations//2].set_xlabel('Unity x axis')

    else:
        for i in range(n_components):
            # each is a matrix of (n_locations, 1)
            component = np.load(f'{results_path}/components_{i+1}.npy')
            ax[i].scatter(
                x_axis_coords, y_axis_coords, 
                c=component, cmap='viridis', s=144)
            ax[i].set_title(f'c{i+1}')
            ax[i].set_xlim(x_min, x_max)
            ax[i].set_ylim(y_min, y_max)
        ax[n_components//2].set_ylabel('Unity z axis')
        ax[-1].set_xlabel('Unity x axis')

    title = f'{movement_mode}, {model_name}, {output_layer}, {reduction_method}'
    plt.tight_layout()
    # plt.suptitle(title, y=1.05)
    plt.savefig(f'{results_path}/components.png')


def plot_combined_components(
        config_version, method
    ):
    """
    Plot individual components of all rotation jointly,
    subject to a combine `method`, in {'add', 'average', ...}.
    Such combine is done per 2D location across all rotations.

    Impl:

    Each row of the subplot is a ranked PC and the first
    column is the combined component over all rotations plotted
    as 2D heatmap; the second column is the distribution (e.g. hist)
    of these combined components.

    Return:

    """
    config = utils.load_config(config_version)
    results_path = utils.return_results_path(config_version)
    
    # map components back onto 2D space
    if config['movement_mode'] == '2d':
        x_axis_coords = []
        y_axis_coords = []
        # same idea as generating the frames in Unity
        # so we get decimal coords in between the grid points
        for i in range(
                config['x_min']*config['multiplier'], 
                config['x_max']*config['multiplier']+1
            ):
            for j in range(
                config['y_min']*config['multiplier'], 
                config['y_max']*config['multiplier']+1
            ):
                x_axis_coords.append(i/config['multiplier'])
                y_axis_coords.append(j/config['multiplier'])
    
    fig, ax = plt.subplots(
        config['n_components'], 2,
        figsize=(5, 2*int(config['n_components']))
    )

    for i in range(config['n_components']):
        # each is a matrix of (n_components, n_locations, n_rotations)
        component = np.load(f'{results_path}/components_{i+1}.npy')
        # rotations of each component is combined by `method`
        # to get a single component per location
        if method == 'add':
            component = np.sum(component, axis=-1)
        elif method == 'average':
            component = np.mean(component, axis=-1)

        # first column plots the combined component
        ax[i, 0].scatter(
            x_axis_coords, y_axis_coords, 
            c=component, cmap='viridis', s=99
        )
        ax[i, 0].set_title(f'c{i+1}')
        ax[i, 0].set_xlim(config['x_min'], config['x_max'])
        ax[i, 0].set_ylim(config['y_min'], config['y_max'])
        
        # second column plots the distribution of the combined component
        ax[i, 1].hist(component, bins=20)
        ax[i, 1].set_title(f'c{i+1} distribution')

    ax[config['n_components']//2, 0].set_ylabel('Unity z axis')
    ax[-1, 0].set_xlabel('Unity x axis')

    # title = f'{config["model_name"]}, {config["output_layer"]}, {method}'
    # plt.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.savefig(f'{results_path}/components_combined_{method}.png')
    

def plot_variance_explained(
        movement_mode,
        model_name,
        output_layer,
        reduction_method,
        results_path,
    ):
    assert reduction_method == 'pca', "only pca supported for now"

    explained_variance_ratio = np.load(
        f'{results_path}/explained_variance_ratio.npy'
    )

    def accumulate_sum(array):
        for i in range(1, len(array)):
            prev = array[i-1]
            now = array[i]
            array[i] = prev + now
        return array

    # need to plot accumulatively
    accumulative_variance = accumulate_sum(explained_variance_ratio)

    # find the number of components that explain 90% of variance
    n_components = 0
    for i in range(len(accumulative_variance)):
        if accumulative_variance[i] >= 0.9:
            n_components = i+1
            break

    fig, ax = plt.subplots()
    ax.scatter(range(len(accumulative_variance)), accumulative_variance)
    ax.set_xlabel('number of components')
    ax.set_ylabel('accumulative variance explained')
    ax.set_xticks(range(0, len(accumulative_variance), 10))
    ax.set_xticklabels(range(1, len(accumulative_variance)+1, 10))
    plt.suptitle(f'{movement_mode}, {model_name}, {output_layer}, {reduction_method}\n{n_components} explained 90% of variance')
    plt.savefig(f'{results_path}/explained_variance_ratio.png')


def plot_env_diff(
        env_i, env_j, 
        movement_mode, 
        model_name, 
        output_layer, 
        reduction_method,
        n_components=9,
    ):
    """
    Given two envs that differ some way, e.g. one 
    has a decorated wall or one has a different lighting,
    we compare the 2D component heatmaps, hoping to better 
    understand what the diff will change the 2D components.
    """
    # i and j are the same configs except for the env (only one diff)
    # so we use i to extract env dimensions are are again same.
    template_config = utils.load_config(
        f'{env_i}_{movement_mode}_{model_name}_' \
        f'{output_layer}_{n_components}_{reduction_method}'
    )
    x_min = template_config['x_min']
    x_max = template_config['x_max']
    y_min = template_config['y_min']
    y_max = template_config['y_max']
    multiplier = template_config['multiplier']

    results_path_env_i = \
        f'results/{env_i}/{movement_mode}/{model_name}' \
        f'/{output_layer}/{reduction_method}'
    
    results_path_env_j = \
        f'results/{env_j}/{movement_mode}/{model_name}' \
        f'/{output_layer}/{reduction_method}'

    # collect the top n components in a list
    # this is for subplotting
    components = []
    for i in range(n_components):
        env_i_components = np.load(
            f'{results_path_env_i}/components_{i+1}.npy'
        )
        env_j_components = np.load(
            f'{results_path_env_j}/components_{i+1}.npy'
        )
        components.append(env_i_components - env_j_components)
        # print(stats.spearmanr(env_i_components, env_j_components))
        # a = np.isclose(
        #     env_i_components, 
        #     env_j_components, 
        #     rtol=0.1, 
        # )
        # print(a)
        # exit()

    # create the subplots (assume square)
    subplot_dim = int(np.sqrt(n_components))
    fig, ax = plt.subplots(subplot_dim, subplot_dim)

    if movement_mode == '1d':
        x_axis_coords = []
        y_axis_coords = np.zeros(len(components[0]))
        for i in range(x_min*multiplier, x_max*multiplier+1):
            x_axis_coords.append(i/multiplier)

    elif movement_mode == '2d':
        x_axis_coords = []
        y_axis_coords = []
        # same idea as generating the frames in Unity
        # so we get decimal coords in between the grid points
        for i in range(x_min*multiplier, x_max*multiplier+1):
            for j in range(y_min*multiplier, y_max*multiplier+1):
                x_axis_coords.append(i/multiplier)
                y_axis_coords.append(j/multiplier)

    # each subplot is a component across
    # the above x and y coords
    for subplot_i in range(n_components):
        components_i = components[subplot_i]
        row_idx = int(subplot_i / subplot_dim)
        col_idx = subplot_i % subplot_dim
        # print(f'row_idx: {row_idx}, col_idx: {col_idx}')

        ax[row_idx, col_idx].set_title(f'component {subplot_i+1}')
        ax[row_idx, col_idx].set_xlim(x_min, x_max)
        ax[row_idx, col_idx].set_ylim(y_min, y_max)
        ax[row_idx, col_idx].scatter(
            x_axis_coords, y_axis_coords, 
            c=components_i, cmap='viridis', s=24
        )

        # turn off unnec axes
        if row_idx < subplot_dim-1:
            ax[row_idx, col_idx].set_xticks([])
        if col_idx > 0:
            ax[row_idx, col_idx].set_yticks([])

    # add axes labels
    ax[1, 0].set_ylabel('Unity z axis')
    ax[2, 1].set_xlabel('Unity x axis')

    # plt.tight_layout()
    plt.suptitle(
        f'{env_i}-{env_j} \n{movement_mode}, ' \
        f'{model_name}, {output_layer}, {reduction_method}')
    results_path = f'results/{env_i}-{env_j}/{movement_mode}/{model_name}' \
                   f'/{output_layer}/{reduction_method}'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    plt.savefig(f'{results_path}/components_diff.png')


if __name__ == "__main__":
    import utils
    config_version = "env33_r24_2d_vgg16_fc2_50_maxvar"
    config = utils.load_config(config_version)

    # plot_components(stats=None, config=config)
    plot_combined_components(config_version, method='add')

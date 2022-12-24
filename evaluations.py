import numpy as np
import matplotlib.pyplot as plt


def plot_components(
        n_components,
        results_path, 
        movement_mode
    ):
    ###########
    # Unity env dimension (Unity z axis in python is now y axis)
    x_min = -4
    x_max = 4
    y_min = -4
    y_max = 4
    ###########

    # collect the top n components in a list
    # this is for subplotting
    components = []
    for i in range(n_components):
        components.append(
            np.load(
                f'{results_path}/components_{i+1}.npy')
            )

    # create the subplots (assume square)
    subplot_dim = int(np.sqrt(n_components))
    fig, ax = plt.subplots(subplot_dim, subplot_dim)

    if movement_mode == '1d':
        # x_axis_coords = np.linspace(x_min, x_max, len(components[0]))
        # y_axis_coords = np.zeros(len(components[0]))
        multiplier = 10
        x_axis_coords = []
        y_axis_coords = np.zeros(len(components[0]))
        for i in range(x_min*multiplier, x_max*multiplier+1):
            x_axis_coords.append(i/multiplier)

    elif movement_mode == '2d':
        multiplier = 2  # for decimal coords
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
    plt.savefig(f'{results_path}/components.png')


def plot_variance_explained(results_path, reduction_method):
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
    fig, ax = plt.subplots()
    ax.scatter(range(len(accumulative_variance)), accumulative_variance)
    ax.set_xlabel('number of components')
    ax.set_ylabel('accumulative variance explained')
    ax.set_xticks(range(0, len(accumulative_variance), 10))
    ax.set_xticklabels(range(1, len(accumulative_variance)+1, 10))
    plt.savefig(f'{results_path}/explained_variance_ratio.png')


if __name__ == "__main__":
    movement_mode = '1d'
    model_name = 'vgg16'
    output_layer = 'fc2'
    reduction_method = 'pca'
    results_path = f'results/{movement_mode}/{model_name}/{output_layer}/{reduction_method}'
    n_components = 9

    plot_components(
        n_components, 
        results_path, 
        movement_mode,
    )

    plot_variance_explained(
        results_path, 
        reduction_method,
    )
        
import numpy as np 
import matplotlib.pyplot as plt
import utils
np.random.seed(999)

"""
From grid cells to place cells
"""

def cell_actv(sigma, n_locations, n_cells, agent_locations, cell_locations, n_grid_centers):
    """
    Each cell activity is a vector sum of multiple gaussians wrt agent locations across
    time. 

    Return a matrix of size (n_locations, n_cells),
    where each column is a cell actv over time.

    n_grid_centers: 
        For each grid cell, specify number of centers it has, which will translate 
        to the number of bumps in 2D map. For each center, gaussian distance wrt agent location 
        is computed (just like place-cell). The final cell actv is the sum over centers aross
        locations.
    """
    print('[Check] computing cell actv..')
    # actv2 = np.empty((n_locations, n_grid_centers, n_cells))
    # for l in range(n_locations):
    #     for m in range(n_grid_centers):
    #         for c in range(n_cells):
    #             actv2[l, m, c] = np.exp(
    #                 -np.sum((agent_locations[l] - cell_locations[c, m])**2) / (2 * sigma**2)
    #             )

    agent_locations = agent_locations.reshape(n_locations, 2, 1, 1)
    print(f'agent_locations.shape = ', agent_locations.shape)
    cell_locations = cell_locations.T.reshape(1, 2, n_grid_centers, n_cells)
    print(f'cell_locations.shape = ', cell_locations.shape)
    actv = np.exp(
        -np.sum( (agent_locations-cell_locations)**2, axis=1 ) / (2 * sigma**2)
    )

    # assert np.mean(actv.flatten() == actv2.flatten()) == 1
    res = np.sum(actv, axis=1)
    print(res.shape)
    return res


def transform_and_plot_cells(actv, agent_locations, transform, input_type):
    """
    Transform place cell actv and plot the cells in 2D space.

    Each cell has actv over locations, each needs to be plotted separate.
    """
    print(f'[Check] transform = {transform}')
    if transform == 'svd':
        actv_transformed = utils.SVD_on_cell_actv(actv)
        subplot_title = 'Prin. comp.'

    elif transform == 'ica':
        actv_transformed = utils.ICA_on_cell_actv(actv)
        subplot_title = 'Ind. comp.'
    
    elif transform == 'nmf':
        actv_transformed = utils.NMF_on_cell_actv(actv)
        subplot_title = 'NMF'

    elif transform == 'none':
        actv_transformed = actv
        subplot_title = f'input {input_type} cell'

    # plotting
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    
    # plot top 9 principle components (grid cells) in 2D space
    # given agent locations
    for i in range(4):
        for j in range(4):
            ax[i, j].scatter(
                agent_locations[:, 0], 
                agent_locations[:, 1], 
                c=actv_transformed[:, i * 4 + j], 
                cmap='viridis',
            )
            ax[i, j].set_title(f'{subplot_title} {i * 4 + j + 1}')

    plt.tight_layout()
    plt.savefig(f'input_{input_type}_cells_{transform}.png')
    return actv_transformed


def execute():
    dim = 2
    n_cells = 1000
    n_locations = 1000
    n_grid_centers = 10
    env_length = 10
    sigma = 0.5
    transform = 'nmf'
    # reverse_transform = 'ica'

    # 1. sample gaussian centers in 2d space
    cell_locations = np.empty((n_cells, n_grid_centers, dim))
    for c in range(n_cells):
        for m in range(n_grid_centers):
            cell_locations[c, m] = np.random.uniform(0, env_length, dim)

    # 2. sample agent locations in 2d space
    agent_locations = np.empty((n_locations, dim))
    for l in range(n_locations):
        agent_locations[l] = np.random.uniform(0, env_length, dim)

    # 3. compute cell activity
    # actv -> (n_locations, n_cells)
    actv = cell_actv(
        sigma=sigma, 
        n_locations=n_locations, 
        n_cells=n_cells, 
        agent_locations=agent_locations, 
        cell_locations=cell_locations,
        n_grid_centers=n_grid_centers
    )

    # 4. perform transformation on grid-cell actv and 
    # and plot transformed cell actv
    actv_transformed = transform_and_plot_cells(
        actv, agent_locations, transform, input_type='grid'
    )


if __name__ == "__main__":
    execute()
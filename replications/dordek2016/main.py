import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
np.random.seed(999)

"""
First just try to figure out how PCA is performed (on which dim and map to 2D)
"""

def cell_actv(sigma, n_locations, n_cells, agent_locations, cell_locations):
    """
    Each cell activity is a gaussian wrt agent location

    Return a matrix of size (n_locations, n_cells),
    where each column is a cell actv over time.
    """
    actv = np.empty((n_locations, n_cells))
    for l in range(n_locations):
        for c in range(n_cells):
            actv[l, c] = np.exp(-np.sum((agent_locations[l] - cell_locations[c])**2) / (2 * sigma**2))
    return actv


# def PCA_for_grid_cells(actv, pca_dim):
#     """
#     Compute covariance matrix of actv and 
#     perform PCA on either time or cells dimension.

#     actv: (n_locations, n_cells)

#     either
#         cov_actv: (n_cells, n_cells)
#     or 
#         cov_actv: (n_locations, n_locations)
#     """
#     if pca_dim == 'loc':
#         cov_actv = np.cov(actv)
#     elif pca_dim == 'cell':
#         cov_actv = np.cov(actv.T)
#     print(f'cov_actv.shape: {cov_actv.shape}')

#     pca = PCA(n_components=cov_actv.shape[1])
#     cov_actv_transformed = pca.fit_transform(cov_actv)
#     print(f'cov_actv_transformed.shape: {cov_actv_transformed.shape}')  # (n_loc, n_loc)
#     return cov_actv_transformed


def SVD_for_grid_cells(actv):
    """Compute SVD of actv

    actv: (n_locations, n_cells) -> (n_locations, n_latent_cells)
    """
    pca = PCA(n_components=actv.shape[1])
    actv_transformed = pca.fit_transform(actv)
    print(f'actv_transformed.shape: {actv_transformed.shape}')
    return actv_transformed


def compute_and_plot_grid_cells(actv, agent_locations):
    """
    Plot grid cells in 2D space.

    Each cell has actv over locations, each needs to be plotted separate.
    """
    # SVD
    actv_transformed = SVD_for_grid_cells(actv)

    # plotting
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    
    # plot top 9 principle components (grid cells) in 2D space
    # given agent locations
    for i in range(3):
        for j in range(3):
            ax[i, j].scatter(
                agent_locations[:, 0], 
                agent_locations[:, 1], 
                c=actv_transformed[:, i * 3 + j], 
                cmap='viridis',
            )
            ax[i, j].set_title(f'Prin. comp. {i * 3 + j + 1}')

    plt.savefig('grid_cells.png')


def plot_place_cells(actv, agent_locations):
    # plotting
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    
    # plot the first 9 place cells in 2D space
    # given agent locations
    for i in range(3):
        for j in range(3):
            ax[i, j].scatter(
                agent_locations[:, 0], 
                agent_locations[:, 1], 
                c=actv[:, i * 3 + j], 
                cmap='viridis',
            )
            ax[i, j].set_title(f'Place cell {i * 3 + j + 1}')

    plt.savefig('place_cells.png')


def execute():
    dim = 2
    n_cells = 1000
    n_locations = 1000
    env_length = 10
    sigma = 1
    pca_dim = 'loc'
    n_sampled_cells = 5  # number of cells to plot

    # 1. sample gaussian ceters in 2d space
    cell_locations = np.empty((n_cells, dim))
    for c in range(n_cells):
        cell_locations[c] = np.random.uniform(0, env_length, dim)

    # 2. sample agent locations in 2d space
    agent_locations = np.empty((n_locations, dim))
    for l in range(n_locations):
        agent_locations[l] = np.random.uniform(0, env_length, dim)

    # 3. compute cell activity
    # actv -> (n_locations, n_cells)
    actv = cell_actv(sigma, n_locations, n_cells, agent_locations, cell_locations)

    # 4. perform PCA and plot grid cells
    compute_and_plot_grid_cells(actv, agent_locations)

    # 5. plot place cells
    plot_place_cells(actv, agent_locations)


if __name__ == "__main__":
    execute()
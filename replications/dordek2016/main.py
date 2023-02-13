import numpy as np 
import matplotlib.pyplot as plt
import utils
np.random.seed(999)


def cell_actv(sigma, n_locations, n_cells, agent_locations, cell_locations):
    """
    Each cell activity is a gaussian wrt agent location

    Return a matrix of size (n_locations, n_cells),
    where each column is a cell actv over time.
    """
    actv = np.empty((n_locations, n_cells))
    for l in range(n_locations):
        for c in range(n_cells):
            actv[l, c] = np.exp(
                -np.sum((agent_locations[l] - cell_locations[c])**2) / (2 * sigma**2)
            )
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
#     

def transform_and_plot_cells(actv, agent_locations, transform, input_type):
    """
    Transform place cell actv and plot the cells in 2D space.

    Each cell has actv over locations, each needs to be plotted separate.
    """
    print(f'transform = {transform}')
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
            ax[i, j].set_title(f'{subplot_title} {i * 3 + j + 1}')

    plt.savefig(f'input_{input_type}_cells_{transform}.png')
    return actv_transformed


def execute():
    dim = 2
    n_cells = 1000
    n_locations = 1000
    env_length = 10
    sigma = 1
    transform = 'nmf'
    reverse_transform = 'ica'

    # 1. sample gaussian centers in 2d space
    cell_locations = np.empty((n_cells, dim))
    for c in range(n_cells):
        cell_locations[c] = np.random.uniform(0, env_length, dim)

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
        cell_locations=cell_locations
    )

    # 4. perform transformation on place-cell actv and 
    # and plot transformed cell actv
    actv_transformed = transform_and_plot_cells(
        actv, agent_locations, transform, input_type='place'
    )

    # 5. converse direction performing transform on the latent cell actv
    # i.e. apply `reverse_transform` on transformed actv
    if transform != 'none':
        actv_transformed = transform_and_plot_cells(
            actv=actv_transformed, 
            agent_locations=agent_locations, 
            transform=reverse_transform, 
            input_type=transform
        )


if __name__ == "__main__":
    execute()
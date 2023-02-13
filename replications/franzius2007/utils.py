from sklearn.decomposition import PCA, FastICA, NMF


def SVD_on_cell_actv(actv):
    """Compute SVD on cell actv

    Depending on the nature of the actv, 
    the actv could be place cell or latent 
    (depending on the transform) cell actv

    actv: (n_locations, n_cells) -> (n_locations, n_latent_cells)
    """
    pca = PCA(n_components=actv.shape[1])
    actv_transformed = pca.fit_transform(actv)
    print(f'actv_transformed.shape: {actv_transformed.shape}')
    return actv_transformed


def ICA_on_cell_actv(actv):
    """Compute ICA of actv

    actv: (n_locations, n_cells) -> (n_locations, n_latent_cells)
    """
    ica = FastICA(n_components=actv.shape[1])
    actv_transformed = ica.fit_transform(actv)
    print(f'actv_transformed.shape: {actv_transformed.shape}')
    return actv_transformed


def NMF_on_cell_actv(actv):
    """Compute NMF of actv

    actv: (n_locations, n_cells) -> (n_locations, n_latent_cells)
    """
    nmf = NMF(n_components=actv.shape[1], max_iter=1000)
    actv_transformed = nmf.fit_transform(actv)
    print(f'actv_transformed.shape: {actv_transformed.shape}')
    return actv_transformed

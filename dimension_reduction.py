import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF, FastICA, KernelPCA


def compute_components(X, reduction_method, reduction_hparams, random_state=999):
    """
    X is a matrix of shape (num_samples, num_features), where
    each sample is a frame captured in Unity and each feature is a 
    unit representation in a CNN layer.
    """
    print(f'X.shape: {X.shape}')
    print(f'n_components: {np.min(X.shape)}')

    if reduction_method == 'pca':
        print(f'running PCA...')
        pca = PCA(n_components=np.min(X.shape), random_state=random_state)
        X_latent = pca.fit_transform(X)
        explained_variance_ratio = pca.explained_variance_ratio_
        return X_latent, explained_variance_ratio, pca
    
    elif reduction_method == 'kpca':
        print(f'running KPCA...')
        print(f'kernel: {reduction_hparams["kernel"]}')
        print(f'degree: {reduction_hparams["degree"]}')
        kpca = KernelPCA(
            n_components=np.min(X.shape),
            kernel=reduction_hparams['kernel'],
            degree=reduction_hparams['degree'],
            random_state=random_state
        )
        X_latent = kpca.fit_transform(X)
        explained_variance_ratio = kpca.lambdas_ / np.sum(kpca.lambdas_)
        return X_latent, explained_variance_ratio
    
    elif reduction_method == 'nmf':
        print(f'running NMF...')
        nmf = NMF(n_components=np.min(X.shape), random_state=random_state)
        X_latent = nmf.fit_transform(X)
        return X_latent, None

    elif reduction_method == 'ica':
        ica = FastICA(n_components=np.min(X.shape), random_state=random_state)
        X_latent = ica.fit_transform(X)
        return X_latent, None

    elif reduction_method == 'avgmax':
        print(f'running avgmax...')
        # sort the matrix columns based on averaged max
        # values of each column
        col_ranks_desc = np.argsort(np.mean(X, axis=0))[::-1]
        print(f'col_ranks_desc.shape: {col_ranks_desc.shape}')
        X_latent = X[:, col_ranks_desc]
        return X_latent, None
    
    elif reduction_method == 'maxvar':
        print(f'running maxvar...')
        # sort the matrix columns based on max variance
        # of each column
        col_ranks_desc = np.argsort(np.var(X, axis=0))[::-1]
        print(f'col_ranks_desc.shape: {col_ranks_desc.shape}')
        X_latent = X[:, col_ranks_desc]
        return X_latent, None
        

def compute_component_matrix(
        X, reduction_method, 
        component_matrix_type='loadings', 
        random_state=999
    ):
    """
    X is a matrix of shape (num_samples, num_features), where
    each sample is a frame captured in Unity and each feature is a 
    unit representation in a CNN layer.
    """
    print(f'X.shape: {X.shape}')
    print(f'n_components: {np.min(X.shape)}')

    if reduction_method == 'pca':
        print(f'running PCA...')
        pca = PCA(n_components=np.min(X.shape), random_state=random_state)
        pca.fit(X)
        Sigma = pca.singular_values_
        Vt = pca.components_
        if component_matrix_type == 'Vt':
            return Vt
        elif component_matrix_type == 'loadings':
            loadings = np.dot(np.diag(Sigma), Vt)
            return loadings
    
    elif reduction_method == 'nmf':
        print(f'running NMF...')
        print('Warning: NMF loadings are not defined and somewhat questionable.')
        print('Warning: NMF loadings are not defined and somewhat questionable.')
        print('Warning: NMF loadings are not defined and somewhat questionable.')
        nmf = NMF(n_components=np.min(X.shape), random_state=random_state)
        nmf.fit(X)
        return nmf.components_
    
    elif reduction_method == 'ica':
        print(f'running ICA...')
        print('Warning: ICA loadings are not defined and somewhat questionable.')
        print('Warning: ICA loadings are not defined and somewhat questionable.')
        print('Warning: ICA loadings are not defined and somewhat questionable.')
        ica = FastICA(n_components=np.min(X.shape), random_state=random_state)
        ica.fit(X)
        return ica.components_


if __name__ == "__main__":
    X = np.random.rand(100, 4)
    # X_latent, explained_variance_ratio = compute_components(X, 'avgmax')
    # print(X_latent.shape)
    # print(explained_variance_ratio)
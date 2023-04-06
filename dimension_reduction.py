import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF, FastICA, KernelPCA


def compute_components(X, reduction_method, reduction_hparams, random_state=999):
    """
    X is a matrix of shape (num_samples, num_features), where
    each sample is a frame captured in Unity and each feature is a 
    unit representation in a CNN layer or raw image pixel.
    """
    print(f'[Check] max n_components: {np.min(X.shape)}')
    if reduction_method == 'pca':
        print(f'[Check] running PCA...')
        pca = PCA(n_components=np.min(X.shape), random_state=random_state)
        X_latent = pca.fit_transform(X)
        explained_variance_ratio = pca.explained_variance_ratio_
        return X_latent, explained_variance_ratio, pca
    
    elif reduction_method == 'kpca':
        print(f'[Check] running KPCA...')
        print(f'  kernel: {reduction_hparams["kernel"]}')
        print(f'  degree: {reduction_hparams["degree"]}')
        kpca = KernelPCA(
            n_components=np.min(X.shape),
            kernel=reduction_hparams['kernel'],
            degree=reduction_hparams['degree'],
            random_state=random_state
        )
        X_latent = kpca.fit_transform(X)
        explained_variance_ratio = kpca.lambdas_ / np.sum(kpca.lambdas_)
        return X_latent, explained_variance_ratio, kpca
    
    elif reduction_method == 'nmf':
        print(f'[Check] running NMF...')
        nmf = NMF(n_components=np.min(X.shape), random_state=random_state)
        X_latent = nmf.fit_transform(X)
        return X_latent, None, nmf

    elif reduction_method == 'ica':
        print(f'[Check] running ICA...')
        ica = FastICA(n_components=np.min(X.shape), random_state=random_state)
        X_latent = ica.fit_transform(X)
        return X_latent, None, ica

    elif reduction_method == 'avgmax':
        print(f'[Check] running avgmax...')
        # sort the matrix columns based on averaged max
        # values of each column
        col_ranks_desc = np.argsort(np.mean(X, axis=0))[::-1]
        print(f'  col_ranks_desc.shape: {col_ranks_desc.shape}')
        X_latent = X[:, col_ranks_desc]
        return X_latent, None, None
    
    elif reduction_method == 'maxvar':
        print(f'[Check] running maxvar...')
        # sort the matrix columns based on max variance
        # of each column
        col_ranks_desc = np.argsort(np.var(X, axis=0))[::-1]
        print(f'  col_ranks_desc.shape: {col_ranks_desc.shape}')
        X_latent = X[:, col_ranks_desc]
        return X_latent, None, None
    
    elif reduction_method == 'minvar':
        print(f'[Check] running minvar...')
        # sort the matrix columns based on min variance
        # of each column
        col_ranks_desc = np.argsort(np.var(X, axis=0))
        print(f'  col_ranks_desc.shape: {col_ranks_desc.shape}')
        X_latent = X[:, col_ranks_desc]
        return X_latent, None, None


if __name__ == "__main__":
    pass
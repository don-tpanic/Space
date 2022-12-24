import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF


def compute_components(X, reduction_method, random_state=999):
    """
    X is a matrix of shape (num_samples, num_features), where
    each sample is a frame captured in Unity and each feature is a 
    unit representation in a CNN layer.
    """
    print(f'X.shape: {X.shape}')
    print(f'n_components: {np.min(X.shape)}')

    if reduction_method == 'pca':
        pca = PCA(n_components=np.min(X.shape), random_state=random_state)
        X_latent = pca.fit_transform(X)
        explained_variance_ratio = pca.explained_variance_ratio_
        return X_latent, explained_variance_ratio
    
    elif reduction_method == 'nmf':
        nmf = NMF(n_components=np.min(X.shape), random_state=random_state)
        X_latent = nmf.fit_transform(X)
        return X_latent, None


if __name__ == "__main__":
    X = np.random.rand(100, 4)
    X_latent, explained_variance_ratio = compute_components(X, 'pca')
    print(X_latent.shape)
    print(explained_variance_ratio)
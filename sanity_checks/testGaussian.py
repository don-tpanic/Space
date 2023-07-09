import numpy as np
import matplotlib.pyplot as plt


def sample_from_1d_gaussian():
    # Create a 1D Gaussian normalized to 1
    X = np.linspace(-5, 5, 100)
    sigma = 1
    mean = 0
    Z = np.exp(-(X-mean)**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))

    # Normalize the Gaussian
    Z = Z / np.sum(Z)
    
    # Sample from the Gaussian
    sample = np.random.choice(X, p=Z)
    return sample


def sample_from_2d_gaussian():
    # Create a 2D Gaussian normalized to 1
    X = np.linspace(-5, 5, 10)
    Y = np.linspace(-5, 5, 10)
    mean = 0
    covariance = np.array([[1, 0], [0, 1]])
    X, Y = np.meshgrid(X, Y)
    Z = np.exp(-((X-mean)**2 + (Y-mean)**2)/(2*covariance[0,0]**2)) / (2*np.pi*covariance[0,0]*covariance[1,1])

    # Normalize the Gaussian
    Z = Z / np.sum(Z)
    print(Z.shape)

    # sample from the 2d Gaussian
    sample = np.random.Generator.choice([X,Y], p=Z)
    print(sample.shape)
    return sample



def plot_gaussian_samples():
    samples = []
    for i in range(1000):
        samples.append(sample_from_1d_gaussian())
    fig, ax = plt.subplots()
    ax.hist(samples)
    plt.savefig('test.png')


plot_gaussian_samples()

sample_from_2d_gaussian()
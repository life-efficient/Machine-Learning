import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
from get_colors import colors

def get_classification_data(sd=3, m=10, n_features=2, n_clusters=2, variant='blobs', noise=0, factor=0.1):
    if variant == 'circles':
        return sklearn.datasets.make_circles(n_samples=m, factor=factor, noise=noise)
    if variant == 'blobs':
        return sklearn.datasets.make_blobs(n_samples=m, n_features=n_features, centers=n_clusters, cluster_std=sd)

import numpy as np
import matplotlib.pyplot as plt

def get_regression_data(m=20): 
    ground_truth_w = 2.3 # slope
    ground_truth_b = -8 #intercept
    # X = np.random.randn(m, 1)*2
    X = np.random.uniform(0, 1, size=(m, 1))*2
    # print(X)
    idxs = np.argsort(X, axis=0)
    idxs = np.squeeze(idxs)
    # print(idxs)
    X = X[idxs]
    # print(X)
    # print(X.shape)
    # X = X[np.argsort(X, axis=0)]
    Y = ground_truth_w*X + ground_truth_b + 0.2*np.random.randn(m, 1)
    # print(X.shape, Y.shape)
    return X, Y #returns X (the input) and Y (labels)

def show_regression_data(X, Y):
    plt.figure()
    plt.scatter(X, Y, c='r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def visualise_regression_data(X, Y, y_hat=None):
    plt.figure()
    plt.scatter(X, Y, c='r', label='Label')
    if y_hat is not None:
        plt.plot(X, y_hat, c='b', label='Hypothesis', marker='x')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def calc_accuracy(predictions, labels):
    return np.mean((predictions == labels).astype(int)) * 100

def visualise_predictions(H, X, Y=None, n=50):
    xmin, xmax, ymin, ymax = min(X[:, 0]), max(X[:, 0]), min(X[:, 1]), max(X[:, 1])
    meshgrid = np.zeros((n, n))
    for x1_idx, x1 in enumerate(np.linspace(xmin, xmax, n)): # for each column
        for x2_idx, x2 in enumerate(np.linspace(ymin, ymax, n)): # for each row
            h = H(np.array([[x1, x2]])).astype(int)[0]
            meshgrid[n-1-x2_idx, x1_idx] = h # axis 0 is the vertical direction starting from the top and increasing downward
    if Y is not None:
        for idx in list(set(Y)):
            plt.scatter(X[Y == idx][:, 0], X[Y== idx][:, 1], c=colors[idx])
    else:
        plt.scatter(X[:,0], X[:, 1])
    plt.imshow(meshgrid, extent=(xmin, xmax, ymin, ymax), cmap='winter')


def show_data(X, Y, predictions=None):
    for i in range(min(Y), max(Y)+1):
        y = Y == i
        x = X[y]
        plt.scatter(x[:, 0], x[:, 1], c=colors[i])
        if predictions is not None:
            y = predictions == i
            x = X[y]
            plt.scatter(x[:, 0], x[:, 1], c=colors[i], marker='x', s=100)
    plt.show()

    
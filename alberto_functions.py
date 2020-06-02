import numpy as np
import plotly
import plotly.graph_objects as go
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gzip
from mlxtend.data import loadlocal_mnist
from sklearn.manifold import TSNE

# 1 Example using plotly dash

def load_usps_data(fname, digit=3):
    ''' Loads USPS (United State Postal Service) data from <fname>
    Definition:  X, Y = load_usps_data(fname, digit = 3)
    Input:       fname   - string
                 digit   - optional, integer between 0 and 9, default is 3
    Output:      X       -  DxN array with N images with D pixels
                 Y       -  1D array of length N of class labels
                                 1 - where picture contains the <digit>
                                -1 - otherwise
    '''
    # load the data
    data = io.loadmat(fname)
    # extract images and labels
    X = data['data_patterns']
    Y = data['data_labels']
    Y = Y[digit, :]

    X_val = np.squeeze(X[:, np.argwhere(Y == 1)])
    return X_val.T, Y

def gammaidx(X, k):
    '''  Compute the gamma values using the k nearest neighborus for the each data point of X
    Definition:  y = gammaidx(X, k)
    Input:       X        - NxD array of N data points with D features
                 k        - int, number of neighbours used
    Output:      y        - Nx1 array, contains the gamma value for each data point
    '''
    N = X.shape[0]
    y = np.zeros(N)
    # Find k nearest neighbour for every data sample and compute its correspondent gamma value
    for i in range(N):
        data = np.tile(X[i, :], (N, 1))
        distance = np.linalg.norm(x=(data - X), axis=1)
        idx = np.argpartition(distance, k)
        # Sort 4 largest distances and add them up
        y[i] = (1/k)*np.sum(distance[idx[0:k+1]])
    return y

def apply_pca(X, m):
    ''' Perform PCA analysis on X and reduce to an m dimensional subspace
    Definition:  y = apply_pca(X, m)
    Input:       X                  - NxD array of N data points with D features
                 m                  - int, dimension of the subspace to project
    Output:      y                  - mxN array, of N data points with m features
                 remained_variance  - float, percentage of remained variance with m choosen components
    '''

    pca = PCA(n_components=m)
    pca.fit(X)
    y = pca.transform(X)
    variance_components = pca.explained_variance_ratio_
    remained_variance = np.sum(variance_components[:m])*100
    return y, remained_variance, pca

def visualize_twod_data(X, n_inlier, n_outliers):
    plt.figure(num=1, figsize=(7.2, 4.45), facecolor='white')
    ax = plt.subplot(111)
    ax.scatter(X[:n_inlier, 0], X[:n_inlier, 1], c='blue', label='inliers')
    ax.scatter(X[n_inlier:, 0], X[n_inlier:, 1], c='red', label='outliers')
    plt.tight_layout()
    plt.legend()

def load_mnsit():
    X, y = loadlocal_mnist(
        images_path='/Users/albertorodriguez/Desktop/Current Courses/InternenServLab/train-images-idx3-ubyte',
        labels_path='/Users/albertorodriguez/Desktop/Current Courses/InternenServLab/train-labels-idx1-ubyte')
    return X, y

def visualize_mnist(X):
    # Randomly pick 9 images
    fig = plt.figure(num='Mnist', figsize=(7.2, 4.45))
    for i, a in enumerate([1,2,3,4,5,6,7,8,9]):
        ax = fig.add_subplot(5, 2, i + 1)
        ax.imshow(X[a, :].reshape((28, 28)))
        ax.set_axis_off()


def select_digit(X,y, digit):
    X_digit = X[np.argwhere(y == digit),:]
    return X_digit

def apply_tsne(X, m, neighbours):
    ''' Perform t-distributed Stochastic Neighbor Embedding. on X and reduce to an m dimensional subspace
    Definition:  y = apply_pca(X, m)
    Input:       X                  - NxD array of N data points with D features
                 m                  - int, dimension of the subspace
    Output:      y                  - mxN array, of N data points with m features
    '''
    y = TSNE(n_components=m, perplexity=neighbours).fit_transform(X)
    return y


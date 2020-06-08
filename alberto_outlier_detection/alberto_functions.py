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
import pandas as pd
import math
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist  # fast distance matrices

# 1 Example using plotly dash

class EvaData():
    def __init__(self):
        self.dataset = ''
        self.pandas_data_frame = None
        self.X = None
        self.Y = None
        self.X_original = None
        self.outl_score = None
        self.d = None
        self.n = None
        self.labels = False
        self.labels_name = None
        self.file_name = None
        self.feature_names = None
        self.outl_scores = None
        self.remained_variance = None


    def load_data(self, file_name):
        '''
        Loads a csv file into a numpy data X and labels y
        Definition:  load_data(self, file_name)
        Input:       fname   - string, file name ending in .csv

        '''
        self.file_name = file_name
        # Check data frame is in the correct .csv format
        if self.file_name.endswith('.csv'):
            # CSV File with the last columns = labels if there are any labels
            self.pandas_data_frame = pd.read_csv(self.file_name)
            X_all = self.pandas_data_frame.to_numpy()
            # Extract nxd data matrix and dimensions
            self.X = X_all[:, :-1].astype('float64')
            self.X_original =self.X
            self.n, self.d = self.X.shape
            # Extract labels (labels should always be the last column of the dataset)
            self.Y = X_all[:, -1]
            print(self.pandas_data_frame)
            # Check if labels actually exist
            if self.Y[0] == 'No':
                self.labels = False
                del self.pandas_data_frame['labels']
            else:
                self.labels = True

            # Save variable names
            self.feature_names = self.pandas_data_frame.keys()

            # Now we've got the nxd data array X and its corresponding labels Y (if NO labels exist,
            #  this is indicated by the self.labels bool instance)

        else:
            print('Error No CSV FILE') # here maybe some dash error widget-message

    def visualize_original_data(self):
        '''
        If the data is 2Dim or 3Dim, it displays the data in a 2D/3D Plot
        Definition:  visualize_original_data(self)
        '''
        if self.d == 1:
            plt.figure(num='Data 1D Visualisation', figsize=(7, 5))
            ax = plt.gca()
            ax.scatter(self.X[:, 0], np.zeros(self.n))

        elif self.d == 2:
            plt.figure(num='Data 2D Visualisation', figsize=(7, 5))
            ax = plt.gca()
            ax.scatter(self.X[:, 0], self.X[:, 1])
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel(self.feature_names[1])
        elif self.d == 3:
            fig = plt.figure(num='Data 3D Visualisation', figsize=(7, 5))
            ax = Axes3D(fig)
            ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2])
            ax.set_xlabel(self.feature_names[0])
            ax.set_xlabel(self.feature_names[1])
            ax.set_zlabel(self.feature_names[2])
        else:
            # If data is more then three dim just plot the first 2 dimensions
            plt.figure(num='Data 2D Visualisation', figsize=(7, 5))
            ax = plt.gca()
            ax.scatter(self.X[:, 0],
                       self.X[:, 1])  # This can be adjusted if the user wants to plot different features
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel(self.feature_names[1])

    def k_ball_index(self, k, scaling_factor):
        '''
        Computes k outlier score value for every data point. kðxÞ is the radius of the smallest ball centered at x
        containing its k nearest neighbors, i.e. the distance between x and its kth nearest neighbor,
        Definition:  calculate_gamma_index(self, k):
        Input:       k              - int, number of nearest neighbours the algorithm uses to compute the outlier score values
                     scaling_factor - int, factor by which the outlier scores are scaled in order to better recognize them
        '''
        ditsance_matrix = cdist(self.X, self.X, 'euclidean')
        self.outl_scores = np.zeros(self.n)

        for i in range(self.n):
            local_dist = ditsance_matrix[i, :]
            local_dist = local_dist[local_dist != 0]
            self.outl_scores[i] = np.sort(local_dist)[k-1]

        # Scale outl scores
        self.outl_scores = self.outl_scores * scaling_factor

        # Plot part
        plt.figure(num='Gamma Index visualisation', figsize=(7, 5))
        ax = plt.gca()
        if self.d == 1:
            ax.scatter(self.X[:, 0],np.zeros(self.n), s=self.outl_scores)

        else:
            ax.scatter(self.X[:, 0], self.X[:, 1],
                       s=self.outl_scores)  # This can be adjusted if the user wants to plot different features
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel(self.feature_names[1])

        # This information may be displayed in the dash board somehow so that the user selects a threshold
        print('Threshold value may be selected between: {} and : {}'.format(np.min(self.outl_scores),
                                                                            np.max(self.outl_scores)))
    def calculate_gamma_index(self, k, scaling_factor):
        ''' Computes Gamma index for the Data matrix K by computing the distances to its k nearest neighbours
            and plots the result in a scatter plot
        Definition:  calculate_gamma_index(self, k):
        Input:       k              - int, number of nearest neighbours the algorithm uses to compute the outlier score values
                     scaling_factor - int, factor by which the outlier scores are scaled in order to better recognize them
        '''

        # Computation part
        self.outl_scores = np.zeros(self.n)
        # Find k nearest neighbour for every data sample and compute its correspondent gamma value
        for i in range(self.n):
            data = np.tile(self.X[i, :], (self.n, 1))
            distance = np.linalg.norm(x=(data - self.X), axis=1)
            idx = np.argpartition(distance, k)
            # Sort 4 largest distances and add them up
            self.outl_scores[i] = (1 / k) * np.sum(distance[idx[0:k + 1]])

        # Scale outlier scores to better recognize
        self.outl_scores = self.outl_scores * scaling_factor

        # Plot part
        plt.figure(num='Gamma Index visualisation', figsize=(7, 5))
        ax = plt.gca()
        if self.d == 1:
            ax.scatter(self.X[:, 0],np.zeros(self.n), s=self.outl_scores)

        else:
            ax.scatter(self.X[:, 0], self.X[:, 1],
                       s=self.outl_scores)  # This can be adjusted if the user wants to plot different features
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel(self.feature_names[1])

        # This information may be displayed in the dash board somehow so that the user selects a threshold
        print('Threshold value may be selected between: {} and : {}'.format(np.min(self.outl_scores),
                                                                            np.max(self.outl_scores)))


    def visualize_outliers(self, threshold):
        '''
        Visualises the outliers of a dataset depending on the method that has been used to compute the outlier scores
        and the threshold the user seleects.

        Definition:  calculate_gamma_index(self, k):
        Input:       method             - str, name of the method used to compute the outlier score values
                     threshold          - float, all data points having an outlier score > threshold are considered
                                          outliers, the rest inliers
        '''
        if self.d == 1:
            plt.figure(num='Outlier visualisation', figsize=(7, 5))
            ax = plt.gca()
            print(self.X[self.outl_scores < threshold, 0].shape[0])
            ax.scatter(self.X[self.outl_scores > threshold, 0], np.zeros(self.X[self.outl_scores > threshold, 0].shape[0]),
                       c='red', label='Outliers')

            ax.scatter(self.X[self.outl_scores <= threshold, 0], np.zeros(self.X[self.outl_scores <= threshold, 0].shape[0]),
                       c='blue', label='Inliers')

        else:
            plt.figure(num='Outlier visualisation', figsize=(7, 5))
            ax = plt.gca()
            ax.scatter(self.X[self.outl_scores > threshold, 0],
                       self.X[self.outl_scores > threshold, 1],
                       c='red', label='Outliers')
            ax.scatter(self.X[self.outl_scores <= threshold, 0],
                       self.X[self.outl_scores <= threshold, 1],
                       c='blue', label='Inliers')

    def apply_PCA(self, m):
        ''' Perform PCA analysis on X and reduce to an m dimensional subspace
        Definition:  apply_pca(X, m)
        Input:       X                  - NxD array of N data points with D features
                     m                  - int, dimension of the subspace to project
        '''

        self.pca = PCA(n_components=m)
        self.pca.fit(self.X)
        # Update X
        self.X = self.pca.transform(self.X)
        variance_components = self.pca.explained_variance_ratio_
        self.remained_variance = np.sum(variance_components[:m]) * 100

        # Update n,d
        self.n, self.d = self.X.shape


    def restore_original_data(self):
        ''' This method restoes X,n,d to the original data
        '''
        self.X = self.X_original
        self.n, self.d = self.X.shape

# Create Class object
ex = EvaData()
# Load data file
ex.load_data('two_gaussians.csv')

# vis original data
ex.visualize_original_data()
# Apply PCA on the data reducing to m=1 dimension
ex.apply_PCA(m=1)
# Visualize original dataset
ex.visualize_original_data()
# Compute outlier score value with gammaindex of k_ball index
ex.calculate_gamma_index(k=14, scaling_factor=100)
#ex.k_ball_index(k=10, scaling_factor=100)
# Visualize outliers for a given threshold
# >. e.g outl_scores = [1.2, 1, 6, 7, 100, 10 ,20, 11, 5.5] and threshold = 20
# All datapoints with an outl_score >20 are considered outliers and the rest inliers

ex.visualize_outliers(threshold=50)

ex.restore_original_data()

plt.show()



# Note: INPUT File must be a csv with the last column being the labels of the data set as in iris.csv
# If the dataset has no labels, then the last column should be called 'labels' with 'No' as entries as follows:

#            0         1 labels
#0    1.204615  0.641784     No
#1    0.134248  0.535008     No
#2   -0.318022  0.577072     No
#3    1.324292  0.474854     No
#4    2.019090  0.459790     No
#..        ...       ...    ...
#195 -1.605268 -0.613308     No
#196  0.265515 -0.341374     No
#197 -0.603661 -0.857383     No
#198 -0.545053 -0.263974     No
#199  0.939915 -0.731172     No
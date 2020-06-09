import  numpy
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist  # fast distance matrices
import pandas as pd
from dash.dependencies import Input, Output
from sklearn.manifold import LocallyLinearEmbedding
import plotly.graph_objects as go
from sklearn.manifold import TSNE

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
        ''' Loads a csv file into a numpy data X and labels y
        Definition:  load_data(self, file_name)
        Input:       fname   - string, file name ending in .csv
        '''
        self.file_name = file_name
        # Check data frame is in the correct .csv format
        if self.file_name.endswith('.csv'):
            # CSV File with the last columns = labels if there are any labels
            self.pandas_data_frame = pd.read_csv(self.file_name)
            X_all = self.pandas_data_frame.to_numpy()
            self.dataset_columns = self.pandas_data_frame.columns
            # Extract nxd data matrix and dimensions
            self.X = X_all[:, :-1].astype('float64')
            self.X_original = self.X
            self.n, self.d = self.X.shape
            self.labels_outlier = self.pandas_data_frame['Classification']
            # Save variable names
            self.feature_names = self.pandas_data_frame.keys()
            # Now we've got the nxd data array X and its corresponding labels Y (if NO labels exist,
            #  this is indicated by the self.labels bool instance)

        else:
            print('Error No CSV FILE') # here maybe some dash error widget-message

    def visualize_original_data(self):
        ''' If the data is 2Dim or 3Dim, it displays the data in a 2D/3D Plot
        Definition:  visualize_original_data(self)
        '''
        if self.d == 1:
            fig = px.scatter(self.pandas_data_frame, x=self.dataset_columns[0], y=np.zeros(self.n),
                             color='Classification')
            return fig

        elif self.d == 2:
            fig = px.scatter(self.pandas_data_frame, x=self.dataset_columns[0], y=self.dataset_columns[1],
                             color='Classification')
            return fig
        elif self.d == 3:
            fig = px.scatter_3d(self.pandas_data_frame, x=self.dataset_columns[0], y=self.dataset_columns[1],
                             z=self.dataset_columns[2], color='Classification')
            return fig
        else:
            # If data is more then three dim just plot the first 2 dimensions
            fig = px.scatter(self.pandas_data_frame, x=self.dataset_columns[0], y=self.dataset_columns[1],
                             color='Classification')
            return fig


    def visualize_reduced_data(self):
        ''' If the data is 1DIM, 2Dim or 3Dim, it displays the data in a 1D/2D/3D Plot
        Definition:  visualize_reduced_data(self)
        '''
        # read columns
        columns_data = self.reduced_pandas_data.columns
        if self.d_red == 1:
            fig = px.scatter(self.reduced_pandas_data, x=columns_data[0], y=np.zeros(self.n),
                             color='Classification')
            return fig

        elif self.d_red == 2:
            fig = px.scatter(self.reduced_pandas_data, x=columns_data[0], y=columns_data[1],
                             color='Classification')
            return fig
        elif self.d_red == 3:
            fig = px.scatter_3d(self.reduced_pandas_data, x=columns_data[0], y=columns_data[1],
                             z=columns_data[2], color='Classification')
            return fig
        else:
            # If data is more then three dim just plot the first 2 dimensions
            fig = px.scatter(self.reduced_pandas_data, x=columns_data[0], y=columns_data[1],
                             color='Classification')
            return fig

    def apply_PCA(self, m):
        ''' Perform PCA analysis on X and reduce to an m dimensional subspace
        Definition:  apply_pca(X, m)
        Input:       X                  - NxD array of N data points with D features
                     m                  - int, dimension of the subspace to project
        '''

        self.pca = PCA(n_components=m)
        self.pca.fit(self.X)
        # Update X
        self.X_red = self.pca.transform(self.X)
        variance_components = self.pca.explained_variance_ratio_
        self.remained_variance = np.sum(variance_components[:m]) * 100

        # Update n,d
        self.d_red = m

        # Build pandas dataframe with PCA reduced data
        self.reduced_pandas_data = pd.DataFrame(data=self.X_red)
        self.reduced_pandas_data['Classification'] = self.pandas_data_frame['Classification']
        self.red_method = 'PCA'

    def apply_LLE(self, m, neighbours=5, neigh_algo=None):
        ''' Perform LLE Locally Linear Embedding¶  on X and reduce to an m dimensional subspace
        Definition:  apply_LLE(X, m)
        Input:       X                  - NxD array of N data points with D features
                     m                  - int, dimension of the subspace to project
        '''
        embedding = LocallyLinearEmbedding(n_components=m, n_neighbors=neighbours, neighbors_algorithm='kd_tree')
        # Update X
        self.X_red = embedding.fit_transform(self.X)
        # Update n,d
        self.d_red = m
        # Build pandas dataframe with LLE reduced data
        self.reduced_pandas_data = pd.DataFrame(data=self.X_red)
        self.reduced_pandas_data['Classification'] = self.pandas_data_frame['Classification']
        self.red_method = 'LLE'

    def apply_TSNE(self, m, perplexity=30):
        ''' Perform t-distributed Stochastic Neighbor Embedding. on X and reduce to an m dimensional subspace
        Definition:  apply_LLE(X, m)
        Input:       X                          - NxD array of N data points with D features
                     m                          - int, dimension of the subspace to project
                     perplexity                 - float, related to the number of nearest neighbors that is used in other
                                                  manifold learning algorithms. Larger datasets usually require a larger
                                                  perplexity. Consider selecting a value between 5 and 50. Different
                                                  values can result in significanlty different results.
        '''
        embedding = TSNE(n_components=m, perplexity=perplexity)
        # Update X
        self.X_red = embedding.fit_transform(self.X)
        # Update n,d
        self.d_red = m
        # Build pandas dataframe with LLE reduced data
        self.reduced_pandas_data = pd.DataFrame(data=self.X_red)
        self.reduced_pandas_data['Classification'] = self.pandas_data_frame['Classification']
        self.red_method = 'TSNE'

    def restore_original_data(self):
        ''' This method restoes X,n,d to the original data
        '''
        self.X = self.X_original
        self.n, self.d = self.X.shape

app = dash.Dash()
app.css.config.serve_locally = False

data_file_name = 'fishbowl_outl.csv'
dim_red_method = 'pca'

main_data = EvaData()
main_data.load_data(file_name=data_file_name)

if dim_red_method == 'pca':
    main_data.apply_PCA(m=2)

fig_original = main_data.visualize_original_data()
fig_reduced = main_data.visualize_reduced_data()

app.layout = html.Div(children=[
    dcc.Graph(id='outliers_red_dim', figure=fig_original),
    dcc.Graph(id='my_scatter_plot', figure=fig_reduced)
])

if __name__ == '__main__':
    app.run_server(debug=True)
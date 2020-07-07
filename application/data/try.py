
import numpy as np
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph
import plotly.graph_objects as go

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
            #print(self.pandas_data_frame.head())
        else:
            print('Error No CSV FILE') # here maybe some dash error widget-message

    def visualize_original_data(self):
        ''' If the data is 2Dim or 3Dim, it displays the data in a 2D/3D Plot
        Definition:  visualize_original_data(self)
        '''
        if self.d == 1:
            fig = px.scatter(self.pandas_data_frame, x=self.dataset_columns[0], y=np.zeros(self.n),
                             color='Classification', title='Not reduced data')
            return fig

        elif self.d == 2:
            fig = px.scatter(self.pandas_data_frame, x=self.dataset_columns[0], y=self.dataset_columns[1],
                             color='Classification', title='Not reduced data')
            return fig
        elif self.d == 3:
            fig = px.scatter_3d(self.pandas_data_frame, x=self.dataset_columns[0], y=self.dataset_columns[1],
                             z=self.dataset_columns[2], color='Classification', title='Not reduced data')
            return fig
        else:
            # If data is more then three dim just plot the first 2 dimensions
            fig = px.scatter(self.pandas_data_frame, x=self.dataset_columns[0], y=self.dataset_columns[1],
                             color='Classification', title='Not reduced data')
            return fig


    def visualize_reduced_data(self, only_outliers = False):
        ''' If the data is 1DIM, 2Dim or 3Dim, it displays the data in a 1D/2D/3D Plot
        Definition:  visualize_reduced_data(self)
        '''
        # read columns
        columns_data = self.reduced_pandas_data.columns
        if self.d_red == 1:
            if only_outliers:
                outl_data_frame = self.reduced_pandas_data[self.reduced_pandas_data['Classification']=='Outlier']
                fig = px.scatter(outl_data_frame, x=columns_data[0], y=np.zeros(outl_data_frame.shape[0]),
                                 color='Classification', title='Reduced data with ' + self.red_method)
            else:
                fig = px.scatter(self.reduced_pandas_data, x=columns_data[0], y=np.zeros(self.n),
                                 color='Classification', title='Reduced data with ' + self.red_method)


        elif self.d_red == 2:
            if only_outliers:
                outl_data_frame = self.reduced_pandas_data[self.reduced_pandas_data['Classification']=='Outlier']

                fig = px.scatter(outl_data_frame, x=columns_data[0], y=columns_data[1],
                                 color='Classification', title='Reduced data with ' + self.red_method)
            else:
                fig = px.scatter(self.reduced_pandas_data, x=columns_data[0], y=columns_data[1],
                             color='Classification', title='Reduced data with '+self.red_method)


        elif self.d_red == 3:
            if only_outliers:
                outl_data_frame = self.reduced_pandas_data[self.reduced_pandas_data['Classification']=='Outlier']
                fig = px.scatter_3d(outl_data_frame, x=columns_data[0], y=columns_data[1],
                                    z=columns_data[2], color='Classification',
                                    title='Reduced data with ' + self.red_method)
            else:
                fig = px.scatter_3d(self.reduced_pandas_data, x=columns_data[0], y=columns_data[1],
                             z=columns_data[2], color='Classification', title='Reduced data with '+self.red_method)

        else:
            if only_outliers:
                outl_data_frame = self.reduced_pandas_data[self.reduced_pandas_data['Classification']=='Outlier']
                print(outl_data_frame)
                fig = px.scatter(outl_data_frame, x=columns_data[0], y=columns_data[1],
                                 color='Classification', title='Reduced data with ' + self.red_method)
            else:
                # If data is more then three dim just plot the first 2 dimensions
                fig = px.scatter(self.reduced_pandas_data, x=columns_data[0], y=columns_data[1],
                             color='Classification', title='Reduced data with '+self.red_method)
        return fig
    def visualize_box_plot_outliers(self, dim):
        columns_data = self.reduced_pandas_data.columns
        fig = px.box(self.reduced_pandas_data, x="Classification", y=columns_data[dim], points="all")

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
        self.remained_variance = np.round(np.sum(variance_components[:m]) * 100, decimals=3)

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
        embedding = LocallyLinearEmbedding(n_components=m, n_neighbors=neighbours, max_iter=100)
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

    def knearest_neighbours(self, n_neighbors, data_name):
        if data_name == 'original':
            # Add one to n_neighbours cause the function takes the point itself as a neighbours resulting in 0 distance
            nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(self.X_original)
            distances, indices = nbrs.kneighbors(self.X_original)
            # Sum distances
            distances_summed = np.sum(distances, axis=1)/n_neighbors
        elif data_name == 'reduced':
            # Add one to n_neighbours cause the function takes the point itself as a neighbours resulting in 0 distance
            nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(self.X_red)
            distances, indices = nbrs.kneighbors(self.X_red)
            # Sum distances
            distances_summed = np.sum(distances, axis=1)/n_neighbors

        # Build pandas data frame
        outlier_indexes = np.argwhere(self.pandas_data_frame['Classification']== 'Outlier').squeeze()
        inlier_indexes = np.argwhere(self.pandas_data_frame['Classification'] == 'Inlier').squeeze()
        distances_inliers = distances_summed[inlier_indexes]
        distances_outliers = distances_summed[outlier_indexes]
        sorted_distances = np.append(distances_inliers, distances_outliers)
        distances_panda = pd.DataFrame(data=sorted_distances)
        distances_panda['Classification'] = self.pandas_data_frame['Classification']

        fig = px.histogram(distances_panda, x=0, color="Classification", title='k-nearest neigbours average distances',
                           nbins=30, marginal="rug", opacity=0.7)


        #
        indices = kneighbors_graph(self.X_red, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
        indices_array = indices.toarray()
        graph = nx.Graph()
        connections = []
        for point in range(self.n):
            all_connections = np.argwhere(indices_array[point, :] == 1)
            for i in all_connections:
                connections.append((point, int(i)))

        nodes_list = [(x, y) for x, y in self.X_red]
        graph.add_edges_from(connections)
        edge_x = []
        edge_y = []

        for edge in graph.edges():
            x0, y0 = nodes_list[edge[0]]
            x1, y1 = nodes_list[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in nodes_list:
            x, y = node
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(graph.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append('# of connections: ' + str(len(adjacencies[1])))

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Network graph',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.show()


    def restore_original_data(self):
        ''' This method restoes X,n,d to the original data
        '''
        self.X = self.X_original
        self.n, self.d = self.X.shape

ex = EvaData()
ex.load_data(file_name='five_gaussians_outl.csv')
ex.apply_PCA(m=2)
ex.knearest_neighbours(n_neighbors=2, data_name='reduced')
#ex.knearest_neighbours(n_neighbors=3, data_name='original')
G = nx.random_geometric_graph(200, 0.125)

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)




import numpy as np
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import Isomap
import umap

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

class VisualizationPlotly():
    ''' This class gives the user the possibility to make different plotly plots to visualize
        an input pandas data frame.
    '''

    def __init__(self, pd_data_frame):
        '''
        Input: pd_data_frame  - pandas data frame representing data
        '''
        self.pd_data_frame = pd_data_frame
        self.pd_data_frame_nolabel = pd_data_frame.drop(columns='Classification')
        # Extract information about the data frame
        self.features = self.pd_data_frame.keys().tolist()

        if 'Classification' in self.features:
            # Read dimensions of the data
            self.d = self.pd_data_frame.shape[1] - 1
            self.features.remove('Classification')
            self.classification = True
        else:
            self.d = self.pd_data_frame.shape[1]
            self.classification = False
        # Read number of samples/examples of the data frame
        self.n = self.pd_data_frame.shape[0]

    def plot_data(self):
        ''' Visualize reduced data in a 1dim,  2dim or 3dim scatter plot. If the panda data frame contains "Classification"
        as one column, the plots are labeled, otherwise not.
        '''

        # If reduced data is just 1 dimensional
        if self.d == 1:
            if self.classification:
                fig = px.scatter(self.pd_data_frame, x=self.features[0], y=np.zeros(self.n),
                                 color='Classification', title='Data', marginal_y='histogram', marginal_x='box',
                                 trendline='lowess')
            else:
                fig = px.scatter(self.pd_data_frame, x=self.features[0], y=np.zeros(self.n), c='blue', title='Data')

        elif self.d == 2:
            if self.classification:
                fig = px.scatter(self.pd_data_frame, x=self.features[0], y=self.features[1],
                                 color='Classification', title='Data', marginal_y='histogram', marginal_x='box',
                                 trendline='lowess')
            else:
                fig = px.scatter(self.pd_data_frame, x=self.features[0], y=self.features[1], c='blue', title='Data')

        else:
            if self.classification:
                fig = px.scatter_3d(self.pd_data_frame, x=self.features[0], y=self.features[1], z=self.features[2],
                                    color='Classification', title='Data')
            else:
                fig = px.scatter_3d(self.pd_data_frame, x=self.features[0], y=self.features[1], z=self.features[2],
                                    color='blue', title='Data')

        return fig

    def box_plot_classifications(self, dim=0):
        ''' This method is only to be used if the pandas dataframe is labeled. It displays the statistical information
        in a boxplot for an input feature (e.g if 2dim data how the inliers/outliers are distributed along the 1 dimension)

        Input: dim    - int, dimension for which the statistical info is shown
        '''
        if self.classification:
            fig = px.box(self.pd_data_frame, x="Classification", y=self.features[dim], points="all",
                         title='Statistical Information')
        else:
            print('Error: Data is not Labeled')
        return fig

    def histogram_data(self, dim):
        ''' Plot an histogram for an given dim/feature of the pandas data frame. If the data is labeled, there is one
            histogram for each class, otherwise not.

            Input: dim    - int, dimension for which the histogram is computed
        '''

        if self.classification:
            fig = px.histogram(self.pd_data_frame, x=self.features[dim], color="Classification", title='Histogram',
                               nbins=30, marginal="rug", opacity=0.7)
        else:
            fig = px.histogram(self.pd_data_frame, x=self.features[dim], title='Histogram', nbins=30, marginal="rug",
                               opacity=0.7)

        return fig

    def graph_neighbours(self, edges, nodes):

        # Check if the graph is 2dim or 3dim
        graph_dim = len(nodes[0])

        if graph_dim == 2:
            edge_x = []
            edge_y = []
            for edge in edges:
                x0, y0 = nodes[edge[0]]
                x1, y1 = nodes[edge[1]]
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
            for node in nodes:
                x, y = node
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    # colorscale options
                    # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                    # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                    # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                    colorscale='YlGnBu',
                    reversescale=True,
                    color=[],
                    size=10,
                    line_width=2))


        elif graph_dim == 3:
            edge_x = []
            edge_y = []
            edge_z = []
            for edge in edges:
                x0, y0, z0 = nodes[edge[0]]
                x1, y1, z1 = nodes[edge[1]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
                edge_z.append(z0)
                edge_z.append(z1)
                edge_z.append(None)


            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []
            node_z = []
            for node in nodes:
                x, y, z = node
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)

            node_trace = go.Scatter3d(
                x=node_x, y=node_y, z = node_z,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    # colorscale options
                    # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                    # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                    # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                    colorscale='YlGnBu',
                    reversescale=True,
                    color=[],
                    size=10,
                    line_width=4))

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

        return fig

    def pie_plot_percentages(self):
        ''' Returns a pie plot chart figure showing how many datapoints % are outliers and how many datapoints are i
        inliers
        '''
        fig = px.pie(self.pd_data_frame, values=self.features[0],
                                     names='Classification', title='Outlier-Percentage Information')
        return fig


    def plot_data_density(self):
        ''' Visualize reduced data in a 1dim,  2dim or 3dim scatter plot. If the panda data frame contains "Classification"
        as one column, the plots are labeled, otherwise not.
        '''

        # If reduced data is just 1 dimensional
        if self.d == 1:
            if self.classification:
                fig = px.density_contour(self.pd_data_frame, x=self.features[0], y=np.zeros(self.n),
                                 color='Classification', title='Density Contour')
            else:
                fig = px.density_contour(self.pd_data_frame, x=self.features[0], y=np.zeros(self.n), c='blue', title='Density Contour')

        else :
            if self.classification:
                fig = px.density_contour(self.pd_data_frame, x=self.features[0], y=self.features[1],
                                 color='Classification', title='Density Contour')
            else:
                fig = px.density_contour(self.pd_data_frame, x=self.features[0], y=self.features[1], c='blue', title='Density Contour')

        return fig

    def plot_dendrogram(self):
        fig = go.Figure(ff.create_dendrogram(self.pd_data_frame_nolabel))
        # fig.update_layout(width=800, height=600)
        fig.update_layout(title="Dendrogram")

        return fig


class DataStatistics():
    def __init__(self):
        self.file_name = None
        self.pandas_dataset = None
        self.X = None
        self.classifications = None
        self.reduced_pandas_dataframe_pca = None
        self.reduced_pandas_dataframe_lle = None
        self.reduced_pandas_dataframe_tsne = None
        self.reduced_pandas_dataframe_kernelpca = None
        self.reduced_pandas_dataframe_isomap = None
        self.reduced_pandas_dataframe_umap = None
        self.features = None
        self.d_red = None
        self.d = None
        self.n = None

    def load_data(self, file_name):
        ''' Load dataset from an input filename (.csv) as a numpy array and as a pandas dataframe. The input csv data
            should have one column named "Classification" with the outliers/inliers labeled. The rest of the columns are
            assumed to be numerical data of the cooresponding dimensions.

        Definition:  load_data(self, file_name)

        Input:       fname   - string, file path name ending in .csv
        '''
        self.file_name = file_name
        # Read CSV file as a pandas data frame
        self.pandas_data_frame = pd.read_csv(file_name)
        # Store Classification labels (outliers or inlier)
        self.classifications = self.pandas_data_frame['Classification']

        # Read features
        self.features = self.pandas_data_frame.keys().tolist()
        self.features.remove('Classification')

        # Pandas Dataframe  without labels in order to perform pca/lle/tsne etc.  on it
        self.pandas_data_frame_nolabels = self.pandas_data_frame[self.features]

        # Read data information (number of features and samples)
        self.n, self.d = self.pandas_data_frame_nolabels.shape

    def apply_pca(self, m=2):
        ''' Apply PCA to the previously loaded pandas data frame in order to reduce the dimensionality of the data
        Definition:  apply_pca(self, m)

        Input:       m  - int, dimension number to which PCA reduces the data

        '''
        self.d_red = m
        # Apply PCA on the unlabeled pandas data frame
        pca = PCA(n_components=m)
        pca.fit(self.pandas_data_frame_nolabels)
        pca_red_data = pca.transform(self.pandas_data_frame_nolabels)  # This is a numpy array
        principalDf = pd.DataFrame(data=pca_red_data)

        # Concadenate the unlabeled pca dataframe with the classifications
        self.reduced_pandas_dataframe_pca = pd.concat([principalDf, self.classifications], axis = 1)

        # Compute how much % of the variance remains (best case 100%)
        variance_components = pca.explained_variance_ratio_
        self.remained_variance = np.round(np.sum(variance_components[:m]) * 100, decimals=3)

    def apply_lle(self, m=2, k=5):
        ''' Perform LLE Locally Linear Embedding  on the dataframe and reduce to an m dimensional subspace
        Definition:  apply_LLE(X, m)
        Input:       m                  - int, dimension of the subspace to project
                     k                  - int, number of the k nearest neighbours the algorithm uses

        '''
        self.d_red = m
        embedding = LocallyLinearEmbedding(n_components=m, n_neighbors=k, max_iter=100)
        # Update X
        lle_red_data = embedding.fit_transform(self.pandas_data_frame_nolabels)
        lleDf = pd.DataFrame(data=lle_red_data)
        # Concadenate the unlabeled pca dataframe with the classifications
        self.reduced_pandas_dataframe_lle = pd.concat([lleDf, self.classifications], axis=1)

    def apply_tsne(self, m=2 , perplexity=30):
        ''' Perform TSNE t-distributed Stochastic Neighbor Embedding on the dataframe and reduce to an m dimensional s
        subspace
        Definition:  apply_tsne(X, m)
        Input:       m                  - int, dimension of the subspace to project
                     perplexity         - float,related to the number of nearest neighbors that is used in other
                                                manifold learning algorithms. Larger datasets usually require a larger
                                                perplexity. Consider selecting a value between 5 and 50. Different
                                                values can result in significanlty different results.

        '''
        self.d_red = m
        embedding = TSNE(n_components=m, perplexity=perplexity)
        # Update X
        tsne_red_data = embedding.fit_transform(self.pandas_data_frame_nolabels)
        tsneDf = pd.DataFrame(data=tsne_red_data)
        # Concadenate the unlabeled pca dataframe with the classifications
        self.reduced_pandas_dataframe_tsne = pd.concat([tsneDf, self.classifications], axis=1)


    def apply_kernelPca(self, m, kernel_type='linear'):
        '''Apply Kernel PCA to the previously loaded pandas data frame in order to reduce the dimensionality of the data
        Definition:  apply_kernelPca(self, m)

        Input:       m              - int, dimension number to which PCA reduces the data
                     kernel_type    - str, type of kernel : “linear” | “poly” | “rbf” | “sigmoid” | “cosine”
        '''
        pcakern = KernelPCA(n_components=m, kernel=kernel_type)
        pcakern.fit(self.pandas_data_frame_nolabels)
        pcak_red_data = pcakern.transform(self.pandas_data_frame_nolabels)
        pcak_Df = pd.DataFrame(data=pcak_red_data)
        # Concadenate the unlabeled kernel pca dataframe with the classifications
        self.reduced_pandas_dataframe_kernelpca = pd.concat([pcak_Df, self.classifications], axis=1)

    def apply_isomap(self, m=2, k=6):
        ''' Perform Isomap  on the dataframe and reduce to an m dimensional subspace with k neighbour
        Definition:  apply_isomap(X, m)
        Input:       m                  - int, dimension of the subspace to project
                     k                  - int, number of the k nearest neighbours the algorithm uses

        '''
        self.d_red = m

        isomap = Isomap(n_neighbors=k, n_components=m)

        # Update X
        isomap_red_data = isomap.fit_transform(self.pandas_data_frame_nolabels)
        isomapDf = pd.DataFrame(data=isomap_red_data)
        # Concadenate the unlabeled pca dataframe with the classifications
        self.reduced_pandas_dataframe_isomap = pd.concat([isomapDf, self.classifications], axis=1)


    def apply_umap(self, m=2, k=5):
        ''' Perform UMAP  on the dataframe and reduce to an m dimensional subspace with k neighbour
        Definition:  apply_umap(X, m)
        Input:       m                  - int, dimension of the subspace to project
                     k                  - int, number of the k nearest neighbours the algorithm uses

        '''
        self.d_red = m

        embedding = umap.UMAP(n_neighbors=k,
                              n_components=m).fit_transform(self.pandas_data_frame_nolabels)
        umap_df = pd.DataFrame(embedding)
        self.reduced_pandas_dataframe_umap = pd.concat([umap_df, self.classifications], axis=1)

    def graph_neighbours(self, n_neighbours, algorithm):
        ''' This method computes the edges and nodes for a given number of neighbours by using the k nearest neighbours,
            the graph is computed for the reduces data -> in 2dim or 3dim

        Input:       n_neighbours    - int, number of neighbours to consider for the knearest algorithm
                     algorithm       - str, refers to the pandas red data frame used to compute the graph nodes+edges
                                       'pca', 'lle', 'tsne', 'kernel_pca', 'isomap', 'umap'
        '''

        # Construct X
        if algorithm == 'pca':
            X = self.reduced_pandas_dataframe_pca
        elif algorithm == 'lle':
            X = self.reduced_pandas_dataframe_lle
        elif algorithm == 'tsne':
            X = self.reduced_pandas_dataframe_tsne
        elif algorithm == 'kernel_pca':
            X = self.reduced_pandas_dataframe_kernelpca
        elif algorithm == 'isomap':
            X = self.reduced_pandas_dataframe_isomap
        elif algorithm == 'umap':
            X = self.reduced_pandas_dataframe_umap

        del X['Classification']
        X = X.to_numpy()

        indices = kneighbors_graph(X, n_neighbors=n_neighbours, mode='connectivity', include_self=False).toarray()
        # Compute the edges and nodes of the graph
        self.edges = []
        for point in range(X.shape[0]):
            all_connections = np.argwhere(indices[point, :] == 1)
            for i in all_connections:
                self.edges.append((point, int(i)))

        if self.d_red == 2:
            self.nodes = [(x, y) for x, y in X]
        elif self.d_red == 3:
            self.nodes = [(x, y, z) for x, y, z in X]

        else:
            print('ERROR DIM to High')


main = DataStatistics()
main.load_data(file_name=r'/Users/albertorodriguez/Desktop/Current Courses/EVA/data/mnist_outl_zero_one.csv')

x = main.pandas_data_frame
print(x[x['Classification'] == 'Outlier'].shape)
#main.apply_lle(m=2, k =30)
#main.apply_pca(m=2)
#print(main.remained_variance)
#main.apply_tsne(m=2)
#main.apply_umap(m=2, k=50)
main.apply_isomap(m=2, k=30)

#vis_pca = VisualizationPlotly(pd_data_frame=main.reduced_pandas_dataframe_pca)
#fig_pca = vis_pca.plot_data()

#vis_tsne = VisualizationPlotly(pd_data_frame=main.reduced_pandas_dataframe_tsne)
#fig_tsne = vis_tsne.plot_data()

#vis_lle = VisualizationPlotly(pd_data_frame=main.reduced_pandas_dataframe_lle)
#fig_lle = vis_lle.plot_data()

vis_isomap = VisualizationPlotly(pd_data_frame=main.reduced_pandas_dataframe_isomap)
fig_isomap = vis_isomap.plot_data()

#vis_umap = VisualizationPlotly(pd_data_frame=main.reduced_pandas_dataframe_umap)
#fig_umap = vis_umap.plot_data()

#fig_pca.show()
#fig_tsne.show()
#fig_umap.show()
#fig_lle.show()
fig_isomap.show()
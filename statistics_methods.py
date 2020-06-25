import numpy as np
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import kneighbors_graph

class DataStatistics():
    def __init__(self):
        self.file_name = None
        self.pandas_dataset = None
        self.X = None
        self.classifications = None
        self.reduced_pandas_dataframe = None
        self.features = None

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

    def apply_pca(self, m):
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
        self.reduced_pandas_dataframe = pd.concat([principalDf, self.classifications], axis = 1)

        # Compute how much % of the variance remains (best case 100%)
        variance_components = pca.explained_variance_ratio_
        self.remained_variance = np.round(np.sum(variance_components[:m]) * 100, decimals=3)

    def apply_lle(self, m, k):
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
        self.reduced_pandas_dataframe = pd.concat([lleDf, self.classifications], axis=1)

    def apply_tsne(self, m , perplexity):
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
        self.reduced_pandas_dataframe = pd.concat([tsneDf, self.classifications], axis=1)


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
        self.reduced_pandas_dataframe = pd.concat([pcak_Df, self.classifications], axis=1)


    def graph_neighbours(self, n_neighbours):
        ''' This method computes the edges and nodes for a given number of neighbours by using the k nearest neighbours,
            the graph is computed for the reduces data -> in 2dim or 3dim

        Input:       n_neighbours    - int, number of neighbours to consider for the knearest algorithm
        '''

        # Construct X
        X = self.reduced_pandas_dataframe
        del X['Classification']
        X = X.to_numpy()

        indices = kneighbors_graph(X, n_neighbors=n_neighbours, mode='connectivity', include_self=False).toarray()
        # Compute the edges and nodes of the graph
        self.edges = []
        for point in range(self.n):
            all_connections = np.argwhere(indices[point, :] == 1)
            for i in all_connections:
                self.edges.append((point, int(i)))

        if self.d_red == 2:
            self.nodes = [(x, y) for x, y in X]
        elif self.d_red == 3:
            self.nodes = [(x, y, z) for x, y, z in X]

        else:
            print('ERROR DIM to High')
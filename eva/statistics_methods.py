import numpy as np
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import Isomap, MDS
import umap
import kmapper as km
from sklearn import preprocessing
from functools import lru_cache


class DataStatistics:
    def __init__(self):
        self.file_name = None
        self.X = None
        self.pandas_data_frame = None
        self.pandas_data_frame_nolabels = None
        self.classifications = None
        self.reduced_pandas_dataframe_pca = None
        self.reduced_pandas_dataframe_lle = None
        self.reduced_pandas_dataframe_tsne = None
        self.reduced_pandas_dataframe_kernelpca = None
        self.reduced_pandas_dataframe_isomap = None
        self.reduced_pandas_dataframe_umap = None
        self.reduced_pandas_dataframe_kmap = None
        self.reduced_pandas_dataframe_mds = None
        self.features = None
        self.d_red = None
        self.d = None
        self.n = None
        self.label_column = None
        self.inliers = None
        self.outliers = None
        self.ratio = None
        self.distances_pd = None

    def load_data(self, file_name, from_file=True, data_frame = None):
        ''' Load dataset from an input filename (.csv) as a numpy array and as a pandas dataframe. The input csv data
            should have one column named "Classification" with the outliers/inliers labeled. The rest of the columns are
            assumed to be numerical data of the cooresponding dimensions.

        Definition:  load_data(self, file_name)

        Input:       fname   - string, file path name ending in .csv
        '''

        if from_file:
            self.file_name = file_name
            # Read CSV file as a pandas data frame
            self.pandas_data_frame = pd.read_csv(file_name)
        else:
            self.pandas_data_frame = data_frame

        self.features = self.pandas_data_frame.keys().tolist()

        # Store Classification labels (outliers or inlier)
        if 'Classification' in self.features :
            self.classifications = self.pandas_data_frame['Classification']
            self.features.remove('Classification')

        # Pandas Dataframe  without labels in order to perform pca/lle/tsne etc.  on it
        self.pandas_data_frame_nolabels = self.pandas_data_frame[self.features]

        # Read data information (number of features and samples)
        self.n, self.d = self.pandas_data_frame_nolabels.shape
        caches = [getattr(self, x) for x in dir(self) if x.startswith('_cached')]
        for cache in caches:
            cache.cache_clear()

    @lru_cache()
    def _cached_pca_transform(self, m):
        pca = PCA(n_components=m)
        pca.fit(self.pandas_data_frame_nolabels)
        pca_red_data = pca.transform(self.pandas_data_frame_nolabels)  # This is a numpy array
        return pd.DataFrame(data=pca_red_data), pca.explained_variance_ratio_

    def apply_pca(self, m=2):
        ''' Apply PCA to the previously loaded pandas data frame in order to reduce the dimensionality of the data
        Definition:  apply_pca(self, m)

        Input:       m  - int, dimension number to which PCA reduces the data

        '''
        self.d_red = m

        principalDf, variance_components = self._cached_pca_transform(m)

        # Concadenate the unlabeled pca dataframe with the classifications
        self.reduced_pandas_dataframe_pca = pd.concat([principalDf, self.classifications], axis = 1)
        # Compute how much % of the variance remains (best case 100%)
        self.remained_variance = np.round(np.sum(variance_components[:m]) * 100, decimals=3)

    @lru_cache()
    def _cached_lle_transform(self, m=2, k=5):
        embedding = LocallyLinearEmbedding(n_components=m, n_neighbors=k, max_iter=100)
        return pd.DataFrame(data=embedding.fit_transform(self.pandas_data_frame_nolabels))

    def apply_lle(self, m=2, k=5):
        ''' Perform LLE Locally Linear Embedding  on the dataframe and reduce to an m dimensional subspace
        Definition:  apply_LLE(X, m)
        Input:       m                  - int, dimension of the subspace to project
                     k                  - int, number of the k nearest neighbours the algorithm uses

        '''
        self.d_red = m
        lleDf = self._cached_lle_transform(m, k)
        # Concadenate the unlabeled pca dataframe with the classifications
        self.reduced_pandas_dataframe_lle = pd.concat([lleDf, self.classifications], axis=1)

    @lru_cache()
    def _cached_tsne_transorm(self, m=2, perplexity=30):
        embedding = TSNE(n_components=m, perplexity=perplexity)
        return pd.DataFrame(data=embedding.fit_transform(self.pandas_data_frame_nolabels))

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
        tsneDf = self._cached_tsne_transorm(m, perplexity)
        # Concadenate the unlabeled pca dataframe with the classifications
        self.reduced_pandas_dataframe_tsne = pd.concat([tsneDf, self.classifications], axis=1)

    @lru_cache()
    def _cached_kernel_pca_tranform(self, m, kernel_type='linear'):
        pcakern = KernelPCA(n_components=m, kernel=kernel_type)
        pcakern.fit(self.pandas_data_frame_nolabels)
        return pd.DataFrame(data=pcakern.transform(self.pandas_data_frame_nolabels))

    def apply_kernelPca(self, m, kernel_type='linear'):
        '''Apply Kernel PCA to the previously loaded pandas data frame in order to reduce the dimensionality of the data
        Definition:  apply_kernelPca(self, m)

        Input:       m              - int, dimension number to which PCA reduces the data
                     kernel_type    - str, type of kernel : “linear” | “poly” | “rbf” | “sigmoid” | “cosine”
        '''
        pcak_Df = self._cached_kernel_pca_tranform(m, kernel_type)
        # Concadenate the unlabeled kernel pca dataframe with the classifications
        self.reduced_pandas_dataframe_kernelpca = pd.concat([pcak_Df, self.classifications], axis=1)

    @lru_cache()
    def _cached_isomap_transform(self, m=2, k=6):
        isomap = Isomap(n_neighbors=k, n_components=m)
        return pd.DataFrame(isomap.fit_transform(self.pandas_data_frame_nolabels))

    def apply_isomap(self, m=2, k=6):
        ''' Perform Isomap  on the dataframe and reduce to an m dimensional subspace with k neighbour
        Definition:  apply_isomap(X, m)
        Input:       m                  - int, dimension of the subspace to project
                     k                  - int, number of the k nearest neighbours the algorithm uses

        '''
        self.d_red = m
        isomapDf = self._cached_isomap_transform(m, k)
        # Concadenate the unlabeled pca dataframe with the classifications
        self.reduced_pandas_dataframe_isomap = pd.concat([isomapDf, self.classifications], axis=1)

    @lru_cache()
    def _cached_umap_transform(self, m=2, k=15):
        embedding = umap.UMAP(n_neighbors=k,
                              n_components=m).fit_transform(self.pandas_data_frame_nolabels)
        return pd.DataFrame(data=embedding)

    def apply_umap(self, m=2, k=15):
        ''' Perform UMAP  on the dataframe and reduce to an m dimensional subspace with k neighbour
        Definition:  apply_umap(X, m)
        Input:       m                  - int, dimension of the subspace to project
                     k                  - int, number of the k nearest neighbours the algorithm uses

        '''
        self.d_red = m
        umap_df = self._cached_umap_transform(m, k)
        self.reduced_pandas_dataframe_umap = pd.concat([umap_df, self.classifications], axis=1)

    @lru_cache()
    def _cached_kmap_transform(self, m=2, k=5, a='PCA'):
        # Initialize
        mapper = km.KeplerMapper(verbose=1)

        # Fit to and transform the data

        if a == 'UMAP':
            projected_data = mapper.fit_transform(self.pandas_data_frame_nolabels, projection=umap.UMAP(n_components=m,
                                                                                                        n_neighbors=k))  # X-Y axis

        if a == 'ISOMAP':
            projected_data = mapper.fit_transform(self.pandas_data_frame_nolabels, projection=Isomap(n_components=m,
                                                                                                     n_neighbors=k))  # X-Y axis

        if a == 'PCA':
            projected_data = mapper.fit_transform(self.pandas_data_frame_nolabels, projection=PCA(n_components=m))

        if a == 'TSNE':
            projected_data = mapper.fit_transform(self.pandas_data_frame_nolabels,
                                                  projection=TSNE(n_components=m, perplexity=30))

        if a == 'LLE':
            projected_data = mapper.fit_transform(self.pandas_data_frame_nolabels,
                                                  projection=LocallyLinearEmbedding(n_components=m, n_neighbors=k,
                                                                                    max_iter=100))
        return pd.DataFrame(projected_data)


    def apply_kmap(self, m=2, k=5, a='PCA'):
        ''' Perform KMAP  on the dataframe and reduce to an m dimensional subspace with k neighbour
        Definition:  KMAP(X, m, a)
        Input:       m                  - int, dimension of the subspace to project
                     k                  - int, number of the k nearest neighbours the algorithm uses
                     a                  - string, algorithm to use


        '''
        self.d_red = m
        kmap_df = self._cached_kmap_transform(m, k, a)
        self.reduced_pandas_dataframe_kmap = pd.concat([kmap_df, self.classifications], axis=1)

    @lru_cache()
    def _cached_mds_transform(self, m=2):
        mds = MDS(n_components=m)
        mds_red_data = mds.fit_transform(self.pandas_data_frame_nolabels)
        return pd.DataFrame(data=mds_red_data)
    def apply_mds(self, m=2):
        self.d_red = m
        mdsDf = self._cached_mds_transform(m)
        # Concadenate the unlabeled pca dataframe with the classifications
        self.reduced_pandas_dataframe_mds = pd.concat([mdsDf, self.classifications], axis=1)


    def graph_neighbours(self, n_neighbours, algorithm):
        ''' This method computes the edges and nodes for a given number of neighbours by using the k nearest neighbours,
            the graph is computed for the reduces data -> in 2dim or 3dim

        Input:       n_neighbours    - int, number of neighbours to consider for the knearest algorithm
                     algorithm       - str, refers to the pandas red data frame used to compute the graph nodes+edges
                                       'pca', 'lle', 'tsne', 'kernel_pca', 'isomap', 'umap', 'kmap'
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
        elif algorithm == 'kmap':
            X = self.reduced_pandas_dataframe_kmap

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


    def create_labeled_df(self):
        if 'Classificationnn' in list(self.pandas_data_frame.keys()):
            pass
        else:
            # Read info from dictionary

            column_name = self.label_column
            selected_inlier = self.inliers
            selected_outlier = self.outliers
            if self.ratio:
                outlier_percentage = self.ratio / 100  # labeld['ratio'] is given in 2%, 80% and we need 0.2, 0.8

            # 2 CASES: Labeled data or not labeled data

            if column_name:
                Inliers_pd = self.pandas_data_frame[self.pandas_data_frame[str(column_name)].isin(selected_inlier)]
                Outliers_pd = self.pandas_data_frame[self.pandas_data_frame[str(column_name)].isin(selected_outlier)]

                # Compute how many Inliers we have and how many Outliers we need to get the selected outliers-ratio(percentage)

                if self.ratio:
                    N_inl = Inliers_pd.shape[0]
                    N_outl = int((N_inl * outlier_percentage) / (1 - outlier_percentage))

                    # Consider the case when our needed N_outl is greater then the given N_outl
                    if Outliers_pd.shape[0] < N_outl:
                        Outliers_pd_final = Outliers_pd
                    else:
                        Outliers_pd_final = Outliers_pd[0:N_outl]

                else:
                    Outliers_pd_final = Outliers_pd

                # Set respective label names to outlier and inliers
                Outliers_pd_final['Classification'] = 'Outliers'
                Inliers_pd['Classification'] = 'Inlier'
                # Just merge/concadenate both inlier and outlier pandas dataframe into the new pd + overwrite
                self.pandas_data_frame = pd.concat([Inliers_pd, Outliers_pd_final], ignore_index=True)

                if column_name == 'Classification':
                    pass
                else:
                    self.pandas_data_frame = self.pandas_data_frame.drop([column_name], axis=1)

                if 'Unnamed: 0' in self.pandas_data_frame.keys().tolist():
                    self.pandas_data_frame = self.pandas_data_frame.drop(['Unnamed: 0'], axis=1)

                self.pandas_data_frame_nolabels = self.pandas_data_frame.drop(['Classification'], axis=1)
                self.classifications = self.pandas_data_frame['Classification']
                # Read data information (number of features and samples)
                self.n, self.d = self.pandas_data_frame_nolabels.shape

            else:
                # In this case we just set all points to Inliers by just adding the classification column
                self.pandas_data_frame['Classification'] = 'Inlier'

    def normalize_data(self):
        ''' This method just normalizes the pandas dataframes to the range 0-1 by using the sklearn function
        MinMaxScaler
        '''
        # Apply MinMaxScaler to the pandas data frame values (nolabels because we do not want 'inliers/outliers') in the
        # values

        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_pd = min_max_scaler.fit_transform(self.pandas_data_frame_nolabels.values)
        # Update pandas dataframes
        self.pandas_data_frame_nolabels= pd.DataFrame(scaled_pd)

        self.pandas_data_frame = pd.DataFrame(scaled_pd)
        self.pandas_data_frame['Classification'] = self.classifications


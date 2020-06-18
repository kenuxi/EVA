import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
from config import iris_config, external_stylesheets
from abc import abstractmethod, ABC
from .assets.layout import html_layout

import dash_daq as daq
import numpy as np
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import networkx as nx
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
        fig = px.box(self.reduced_pandas_data, x="Classification", y=columns_data[dim], points="all",
                     title='Statistical Information for a dimension')

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

        return fig

    def two_dim_graph(self, n_neighbours):
        indices = kneighbors_graph(self.X_red, n_neighbors=n_neighbours, mode='connectivity', include_self=False)
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

        fig_graph = go.Figure(data=[edge_trace, node_trace],
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
        return fig_graph

    def three_dim_graph(self, n_neighbors):
        N = self.n
        indices = kneighbors_graph(self.X_red, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
        indices_array = indices.toarray()
        graph = nx.Graph()
        connections = []
        for point in range(self.n):
            all_connections = np.argwhere(indices_array[point, :] == 1)
            for i in all_connections:
                connections.append((point, int(i)))

        nodes_list = [(x, y, z) for x, y, z in self.X_red]
        graph.add_edges_from(connections)

        Edges = graph.edges

        Xn = [self.X_red[k, 0] for k in range(N)]  # x-coordinates of nodes
        Yn = [self.X_red[k, 1] for k in range(N)]  # y-coordinates
        Zn = [self.X_red[k, 2] for k in range(N)]  # z-coordinates
        Xe = []
        Ye = []
        Ze = []
        for e in Edges:
            Xe += [self.X_red[e[0], 0], self.X_red[e[1], 0], None]  # x-coordinates of edge ends
            Ye += [self.X_red[e[0], 1], self.X_red[e[1], 1], None]  # y-coordinates of edge ends
            Ze += [self.X_red[e[0], 2], self.X_red[e[1], 2], None]  # z-coordinates of edge ends

        trace1 = go.Scatter3d(x=Xe,
                              y=Ye,
                              z=Ze,
                              mode='lines',
                              line=dict(color='rgb(125,125,125)', width=1),
                              hoverinfo='none'
                              )

        trace2 = go.Scatter3d(x=Xn,
                              y=Yn,
                              z=Zn,
                              mode='markers',
                              name='actors',
                              marker=dict(symbol='circle',
                                          size=6,
                                          colorscale='Viridis',
                                          line=dict(color='rgb(50,50,50)', width=0.5)
                                          ),

                              hoverinfo='text'
                              )

        axis = dict(showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title=''
                    )

        layout = go.Layout(
            title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
            width=800,
            height=400,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            margin=dict(
                t=100
            ),
            hovermode='closest',
            annotations=[
                dict(
                    showarrow=False,
                    text="Data source: <a href='http://bost.ocks.org/mike/miserables/miserables.json'>[1] miserables.json</a>",
                    xref='paper',
                    yref='paper',
                    x=0,
                    y=0.1,
                    xanchor='left',
                    yanchor='bottom',
                    font=dict(
                        size=14
                    )
                )
            ], )

        data = [trace1, trace2]
        fig = go.Figure(data=data, layout=layout)

        return fig
    def restore_original_data(self):
        ''' This method restoes X,n,d to the original data
        '''
        self.X = self.X_original
        self.n, self.d = self.X.shape

class Dashboard(ABC):
    def __init__(self, server: flask.Flask, stylesheets: list, prefix='/dash'):
        self.server = server
        self.stylesheets = stylesheets
        self.prefix = prefix
        self.dash_app = None

    @abstractmethod
    def create_dashboard(self, target_column):
        pass

    @abstractmethod
    def init_callbacks(self, target_column):
        pass

class RemoteCSVDashboard(Dashboard):
    def __init__(self, server, stylesheets, prefix, location):
        super().__init__(server, stylesheets, prefix)
        self.df = pd.read_csv(location)

    def create_dashboard(self, target_column):
        pass

    def init_callbacks(self, target_column):
        pass


# Just for the example purpose!

def iris_feature_selector(default_value, features, component_id):
    return dcc.Dropdown(
            id=component_id,
            options=[{'label': feat, 'value': feat} for feat in features],
            value=default_value  # default value
            )


class IrisDashboard(RemoteCSVDashboard):
    def __init__(self, server, stylesheets=external_stylesheets, prefix='/dashboard/', location=iris_config['location']):
        super().__init__(server, stylesheets, prefix, location)

        self.dash_app = dash.Dash(__name__, server=self.server,
                                  routes_pathname_prefix=self.prefix,
                                  external_stylesheets=self.stylesheets)


    def create_dashboard(self, data_dict):

        data_file_name = data_dict['location']# '/Users/albertorodriguez/Desktop/Current Courses/InternenServLab/EVA-merge_flask_dash/application/data/fishbowl_outl.csv'
        dim_red_method = data_dict['algorithm']

        main_data = EvaData()
        main_data.config = data_dict
        main_data.load_data(file_name=data_file_name)
        fig_original = main_data.visualize_original_data()

        fig_percentage_info = px.pie(main_data.pandas_data_frame, values=main_data.dataset_columns[0],
                                     names='Classification', title='Outlier-Percentage Information')

        '''
        if dim_red_method == 'PCA':
            main_data.apply_PCA(m=2)
        elif dim_red_method == 'T-SNE':
            main_data.apply_TSNE(m=2, perplexity=30)

        elif dim_red_method == 'LLE':
            main_data.apply_LLE(m=2, neighbours=5)

        fig_original = main_data.visualize_original_data()
        fig_reduced = main_data.visualize_reduced_data()
        '''

        self.dash_app.css.config.serve_locally = False
        # Boostrap CSS.
        self.dash_app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})  # noqa: E501
        self.dash_app.index_string = html_layout

        self.dash_app.layout = html.Div(
            html.Div([
        html.Div(
            [
                html.H1(children='EVA',
                        className='nine columns',
                        style={
                            'color': '#111',
                            'font-family': 'sans-serif',
                            'font-size': 70,
                            'font-weight': 200,
                            'line-height': 58,
                             'margin': 0,
                             'backgroundColor': '#DCDCDC'
                              },
                        ),
                html.Img(
                    src="https://creativeoverflow.net/wp-content/uploads/2012/03/17-farmhouse.jpg",
                    className='three columns',
                    style={
                        'height': '6%',
                        'width': '6%',
                        'float': 'right',
                        'position': 'right',
                        'margin-top': 0,
                        'margin-right': 10,
                    },
                ),
                html.Div(children='''
                        A visualisation tool to detect outliers
                        ''',
                         style={
                             'color': '#111',
                             'font-family': 'sans-serif',
                             'font-size': 20,
                             'margin': 0,
                             'backgroundColor':'#DCDCDC'
                         },
                        className='nine columns'
                )
            ], className="row"
        ),

        html.Br(),

        html.Div(
            [

            html.Div(children='''
                Data and PCA Information
                ''',
                    style={
                      'color': '#111',
                      'font-family': 'sans-serif',
                      'font-size': 30,
                      'margin': 0
                         },
                         className='three columns'
                         ),
            ], className="row"
        ),

        html.Div(
            [
                html.Div(children='''
                        Data samples N: ''',
                         style={
                             'color': '#111',
                             'font-family': 'sans-serif',
                             'font-size': 17,
                             'margin': 0,
                             'backgroundColor':'#DCDCDC'
                         },
                        className='one column'
                ),

                html.Div([
                    dcc.Input(
                        id="data_samples",
                        type='value',
                        value=main_data.n,
                        placeholder="N%",
                        readOnly=True,
                        size=10,
                        style={
                            'color': '#111',
                            'font-family': 'sans-serif',
                            'font-size': 17,
                            'margin': 0,
                            'backgroundColor': '#DCDCDC'
                        },
                    )

                ], className='one column'),

            ], className="row"
        ),

        html.Div(
            [
                html.Div(children='''
                        Number of features d: ''',
                         style={
                             'color': '#111',
                             'font-family': 'sans-serif',
                             'font-size': 17,
                             'margin': 0,
                             'backgroundColor':'#DCDCDC'
                         },
                        className='one column'
                ),

                html.Div([
                    dcc.Input(
                        id="data_features",
                        type='value',
                        value=main_data.d,
                        placeholder="N%",
                        readOnly=True,
                        size=10,
                        style={
                            'color': '#111',
                            'font-family': 'sans-serif',
                            'font-size': 17,
                            'margin': 0,
                            'backgroundColor': '#DCDCDC'
                        },
                    )

                ], className='one column'),

            ], className="row"
        )

        ,
        html.Div(
            [
                html.Div(children='''
                        Remained Variance %: ''',
                         style={
                             'color': '#111',
                             'font-family': 'sans-serif',
                             'font-size': 17,
                             'margin': 0,
                             'backgroundColor':'#DCDCDC'
                         },
                        className='one column'
                ),

                html.Div([
                    dcc.Input(
                        id="remained_pca_variance",
                        type='value',
                        placeholder="Remained Variance %",
                        readOnly=True,
                        size=10,
                        style={
                            'color': '#111',
                            'font-family': 'sans-serif',
                            'font-size': 17,
                            'margin': 0,
                            'backgroundColor': '#DCDCDC'
                        },
                    )

                ], className='one column'),

            ], className="row"
        ),

        html.Div(
            [
                html.Div([
                    dcc.Graph(id='percentage_outliers', figure=fig_percentage_info),
                ], className='two columns'
                ),
            ], className="row"
        ),


        html.Div(
            [
            html.Div([
                    dcc.Graph(id='reduced_data_plot', figure={})
                ], className='five columns'
                ),
            html.Div([
                    dcc.Graph(id='box_outliers_plot', figure={})
                ], className='five columns'
                )
            ], className="row"
        ),


        html.Div([
            html.Div([
                daq.NumericInput(
                    id='ndim_input',
                    min=1,
                    max=1000,
                    size=120,
                    label='subspace dimension',
                    labelPosition='bottom',
                    value=2),
                html.Div(id='ndim_value')
            ], className='two columns'),

            html.Div([
                dcc.Checklist(
                    id='outlier_only_options',
                    options=[
                        {'label': 'Only show Outliers', 'value': 'yes'}
                    ],
                ),
            ], className='two columns'),

            html.Div([
                daq.NumericInput(
                    id='box_dim_input',
                    min=0,
                    max=100,
                    size=120,
                    label='box dim',
                    labelPosition='bottom',
                    value=0),
            ], className='two columns'),


        ],
            style={'width': '68%', 'display': 'inline-block'},
            className="row"
        ),
        html.Br(),
        html.Div(
            [
                html.Div(children='''
                        Nearest Neighbours
                        ''',
                         style={
                             'color': '#111',
                             'font-family': 'sans-serif',
                             'font-size': 30,
                             'margin': 0
                         },
                        className='nine columns'
                )
            ], className="row"
        ),

        html.Div(
            [
            html.Div([
                dcc.Graph(id='neighbours_plot', figure={}),
                ], className= 'five columns'
                ),
            html.Div([
                    dcc.Graph(id='connected_graph_figure', figure={}),
                ], className='five columns'
                ),
            ], className="row"
        ),

        html.Div([
            html.Div([
                daq.NumericInput(
                    id='neighbours_input',
                    min=1,
                    max=1000,
                    size=120,
                    label='k-neighbours',
                    labelPosition='bottom',
                    value=2),
            ], className='two columns'),

        ],
            style={'width': '68%', 'display': 'inline-block'},
            className="row"
        ),

            ], className='twelve columns offset-by-one')
        )

        @self.dash_app.callback(
            [Output(component_id='reduced_data_plot', component_property='figure'),
             dash.dependencies.Output('remained_pca_variance', 'value'),
             Output(component_id='box_outliers_plot', component_property='figure'),
             Output(component_id='neighbours_plot', component_property='figure'),
             Output(component_id='connected_graph_figure', component_property='figure')],
            [dash.dependencies.Input('ndim_input', 'value'),
             Input('outlier_only_options', 'value'),
             dash.dependencies.Input('box_dim_input', 'value'),
             dash.dependencies.Input('neighbours_input', 'value')]
        )
        def update_graph(m, outl_display_option, box_dim, k_neighbours):
            print(outl_display_option)
            if dim_red_method == 'PCA':
                main_data.apply_PCA(m=m)

            elif dim_red_method == 'LLE':
                main_data.apply_LLE(m=m, neighbours=30)

            elif dim_red_method == 'T-SNE':
                main_data.apply_TSNE(m=m, perplexity=30)


            if outl_display_option == ['yes']:
                fig = main_data.visualize_reduced_data(only_outliers=True)
            else:
                fig = main_data.visualize_reduced_data(only_outliers=False)

            # Box plot for the input dimension
            box_fig = main_data.visualize_box_plot_outliers(dim=box_dim)

            # K nearest neighbours distance histogram on the reduced/lower dimensional dataset
            kn_histogram_fig = main_data.knearest_neighbours(n_neighbors=k_neighbours, data_name='reduced')

            if m == 2:
                graph_figure = main_data.two_dim_graph(n_neighbours=k_neighbours)

            elif m == 3:
                graph_figure = main_data.three_dim_graph(n_neighbors=k_neighbours)

            else:
                graph_figure = {}

            return [fig, main_data.remained_variance, box_fig, kn_histogram_fig, graph_figure]



        return self.dash_app.server


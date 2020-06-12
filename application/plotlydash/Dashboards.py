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
            html.Div([
                dcc.Graph(id='original_data_plot', figure=fig_original),
                ], className= 'five columns'
                ),
            html.Div([
                    dcc.Graph(id='percentage_outliers', figure=fig_percentage_info),
                ], className='five columns'
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
                        className='nine columns'
                )
            ], className="row"
        ),

        html.Br(),

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





            ], className='twelve columns offset-by-one')
        )

        @self.dash_app.callback(
            [Output(component_id='reduced_data_plot', component_property='figure'),
             dash.dependencies.Output('remained_pca_variance', 'value'),
             Output(component_id='box_outliers_plot', component_property='figure')],
            [dash.dependencies.Input('ndim_input', 'value'),
             Input('outlier_only_options', 'value'),
             dash.dependencies.Input('box_dim_input', 'value'),]
        )
        def update_graph(m, outl_display_option, box_dim):
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

            box_fig = main_data.visualize_box_plot_outliers(dim=box_dim)

            return [fig, main_data.remained_variance, box_fig]



        return self.dash_app.server

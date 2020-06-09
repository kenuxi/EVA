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
# Boostrap CSS.
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})  # noqa: E501

app.layout = html.Div(
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
                        'height': '15%',
                        'width': '15%',
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

        html.Div([
            html.Div([
                dcc.Store(id='eva_object', data=''),
                dcc.Dropdown(id='select dataset',
                             options=[
                                 {"label": "Two Gaussians", "value": 'two_gaussians_outl.csv'},
                                 {"label": "Five Gaussians", "value": 'five_gaussians_outl.csv'},
                                 {"label": "Lab Three Dim", "value": 'lab_data_outl.csv'},
                                 {"label": "Fishbowl Outliers", "value": 'fishbowl_outl.csv'},
                                 {"label": "Heart Failure", "value": 'heart_failure.csv'},
                                 {"label": "Shuttle", "value": 'shuttle.csv'}],
                             multi=False,
                             value='',  # initial value
                             style={'width': '98%'},
                             placeholder='Select a dataset'
                             )], className='three columns'),

            html.Div([
                dcc.Dropdown(id='outlier detection method',
                             options=[
                                 {"label": "Gamma index", "value": 'gamma_index'},
                                 {"label": "Epsilon Ball", "value": 'eps_ball'}],
                             multi=False,
                             value='',  # initial value
                             style={'width': '98%'},
                             placeholder='Select Outlier Method'
                             ),


                 ], className='three columns'),

            html.Div([
                daq.NumericInput(
                    id='k_input',
                    min=1,
                    max=1000,
                    size=120,
                    label= 'k value',
                    labelPosition='bottom',
                    value=2),
                html.Div(id='k_value')
            ], className='three columns'),

            html.Div([
                daq.NumericInput(
                    id='threshold_input',
                    min=0,
                    max=1000,
                    size=120,
                    label='threshold value',
                    labelPosition='bottom',
                    value=0),
                html.Div(id='threshold_value')
            ], className='three columns'),


        ],
            style={'width': '68%', 'display': 'inline-block'},
            className="row"
        ),

        html.Div(
            [
            html.Div([
                dcc.Graph(id='my_scatter_plot', figure={})
                ], className= 'six columns'
                ),

                html.Div([
                dcc.Graph(id='outliers_scaled', figure={})
                ], className= 'six columns'
                )
            ], className="row"
        ),
        html.Br(),
        html.Div([
            html.Div(children= 'Dimensionality Reduction',
                     style={
                         'color': '#696969',
                         'font-family': 'sans-serif',
                         'font-size': 30,
                         'font-weight': 100,
                         'line-height': 58,
                         'margin': 0,
                     },
                     className= 'six columns'
                     )

        ], className='row'),

        html.Div([
            html.Div([
                dcc.Store(id='selected_reddim_method', data=''),
                dcc.Dropdown(id='select red method',
                             options=[
                                 {"label": "PCA", "value": 'pca'},
                                 {"label": "LLE", "value": 'lle'},
                                 {"label": "TSN", "value": 'tsne'}],
                             multi=False,
                             value='',  # initial value
                             style={'width': '99%'},
                             placeholder='Select dim reduction'
                             )], className='two columns'),


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
                dcc.Dropdown(id='select outlier method reduced',
                             options=[
                                 {"label": "Gamma index", "value": 'gamma_index'},
                                 {"label": "Epsilon Ball", "value": 'eps_ball'}],
                             multi=False,
                             value='',  # initial value
                             style={'width': '95%'},
                             placeholder='Select Outlier Computation Method'
                             ),

            ], className='two columns'),

            html.Div([
                daq.NumericInput(
                    id='k_input_rdim',
                    min=1,
                    max=1000,
                    size=120,
                    label='k value',
                    labelPosition='bottom',
                    value=2),
                html.Div(id='k_value_rdim')
            ], className='two columns'),

            html.Div([
                daq.NumericInput(
                    id='threshold_input_rdim',
                    min=0,
                    max=1000,
                    size=120,
                    label='threshold value',
                    labelPosition='bottom',
                    value=0),
                html.Div(id='threshold_value_rdim')
            ], className='two columns'),

            html.Div([
                daq.NumericInput(
                    id='lle_neighbours',
                    min=0,
                    max=10000,
                    size=120,
                    label='LLE-neighbours',
                    labelPosition='bottom',
                    value=5),
            ], className='two columns')

        ],
            style={'width': '68%', 'display': 'inline-block'},
            className="row"
        ),

            html.Div(
                [
                html.Div([
                    dcc.Graph(id='red_dim_plot', figure={})
                ], className='six columns'
                ),

                html.Div([
                    dcc.Graph(id='outliers_red_dim', figure={})
                ], className='six columns'
                )
            ], className="row"
        ),
    ], className='eleven columns offset-by-one')
)

# --------------------------------------------------------------
# Callback to load the dataframe and display it
# Callback to load the dataframe and display it
@app.callback(
    [Output(component_id='my_scatter_plot', component_property='figure'),
     Output(component_id='eva_object', component_property='data')],
    [Input(component_id='select dataset', component_property='value')]
)

def update_graph(selected_dataset):
    if selected_dataset is not '':
        # Init class object
        eva_data = EvaData()
        eva_data.load_data(file_name=selected_dataset)
        # Return scatter figure
        fig = eva_data.visualize_original_data()

        return fig, selected_dataset
    else:
        return dash.no_update


# Callback to apply PCA, LLE or SLLE to the dataset that has been previously selected and display it
@app.callback(
    [Output(component_id='red_dim_plot', component_property='figure')],
    [Input(component_id='eva_object', component_property='data'),
     Input(component_id='select red method', component_property='value'),
     dash.dependencies.Input('ndim_input', 'value'),
     dash.dependencies.Input('lle_neighbours', 'value')]
)

def update_graph(selected_dataset, selected_red_dim_method, n_dim, lle_neighbours):
    if selected_dataset is not '':
        # Init class object
        eva_data = EvaData()
        eva_data.load_data(file_name=selected_dataset)
        # Return scatter figure
        if selected_red_dim_method == 'pca':
            eva_data.apply_PCA(m=n_dim)
        elif selected_red_dim_method == 'lle':
            eva_data.apply_LLE(m=n_dim, neighbours=lle_neighbours)
        elif selected_red_dim_method == 'tsne':
            eva_data.apply_TSNE(m=n_dim)

        fig = eva_data.visualize_reduced_data()
        return [fig]
    else:
        return dash.no_update




if __name__ == '__main__':
    app.run_server(debug=True)
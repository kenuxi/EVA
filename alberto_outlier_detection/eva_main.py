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
from utils import get_files_dict

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
            # Extract nxd data matrix and dimensions
            self.X = X_all[:, :-1].astype('float64')
            self.X_original =self.X
            self.n, self.d = self.X.shape
            # Extract labels (labels should always be the last column of the dataset)
            self.Y = X_all[:, -1]
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
        ''' If the data is 2Dim or 3Dim, it displays the data in a 2D/3D Plot
        Definition:  visualize_original_data(self)
        '''
        if self.d == 1:
            fig = px.scatter(x=self.X[:, 0], y=np.zeros(self.n))
            return fig
        elif self.d == 2:
            fig = px.scatter(x=self.X[:, 0], y=self.X[:, 1], title='Original data')
            return fig
        elif self.d == 3:
            fig = px.scatter_3d(x=self.X[:, 0], y=self.X[:, 1], z=self.X[:, 2], title='Original data')
            return fig
        else:
            # If data is more then three dim just plot the first 2 dimensions
            fig = px.scatter(x=self.X[:, 0], y=self.X[:, 1], title='Original data')
            return fig

    def k_ball_index(self, k, scaling_factor):
        ''' Computes k outlier score value for every data point. kðxÞ is the radius of the smallest ball centered at x
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

        # This information may be displayed in the dash board somehow so that the user selects a threshold
        print('Threshold value may be selected between: {} and : {}'.format(np.min(self.outl_scores),
                                                                            np.max(self.outl_scores)))
        # Plot part
        if self.d == 1:
            fig = px.scatter(x=self.X[:, 0], y=np.zeros(self.n), size=self.outl_scores, title='K-Ball-Index-Method for Outlier scores')
            return fig

        elif self.d == 3:
            fig = px.scatter_3d(x=self.X[:, 0], y=self.X[:, 1], z=self.X[:, 2], size=self.outl_scores, title='K-Ball-Index-Method for Outlier scores')
            return fig
        else:
            fig = px.scatter(x=self.X[:, 0], y=self.X[:, 1], size=self.outl_scores, title='K-Ball-Index-Method for Outlier scores')
            return fig


    def gamma_index(self, k, scaling_factor):
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

        # This information may be displayed in the dash board somehow so that the user selects a threshold
        print('Threshold value may be selected between: {} and : {}'.format(np.min(self.outl_scores),
                                                                            np.max(self.outl_scores)))
        # Plot part
        if self.d == 1:
            fig = px.scatter(x=self.X[:, 0], y=np.zeros(self.n), size=self.outl_scores, title='Gamma-Index-Method for Outlier scores')
            return fig

        elif self.d == 3:
            fig = px.scatter_3d(x=self.X[:, 0], y=self.X[:, 1], z=self.X[:, 2], size=self.outl_scores, title='Gamma-Index-Method for Outlier scores')
            return fig
        else:
            fig = px.scatter(x=self.X[:, 0], y=self.X[:, 1], size=self.outl_scores, title='Gamma-Index-Method for Outlier scores')
            return fig


    def visualize_outliers(self, threshold):
        '''
        Visualises the outliers of a dataset depending on the method that has been used to compute the outlier scores
        and the threshold the user seleects.
        Definition:  calculate_gamma_index(self, k):
        Input:       method             - str, name of the method used to compute the outlier score values
                     threshold          - float, all data points having an outlier score > threshold are considered
                                          outliers, the rest inliers
        '''
        n_inliers = len(np.argwhere(self.outl_scores <= threshold))
        n_outliers = len(np.argwhere(self.outl_scores > threshold))
        x_all = np.zeros((self.n, self.d))

        if (n_outliers > 0) and (n_inliers > 0):
            x_all[:n_inliers] = self.X[self.outl_scores <= threshold, :]
            x_all[n_inliers:] = self.X[self.outl_scores > threshold, :]
            labels_inl = ['Inliers'] * n_inliers
            labels_outl = ['Outliers'] * n_outliers

            labels = labels_inl + labels_outl

        elif n_inliers > 0:
            x_all = self.X
            labels = ['Inliers'] * self.n
        else:
            x_all = self.X
            labels = ['Outliers'] * self.n
        if self.d == 1:
            # Construct pandas data frame with inliers as one column and outliers as the other column
            dataset = pd.DataFrame({'x_1': x_all[:, 0],'x_2': np.zeros(self.n),'Classification': labels})
            fig = px.scatter(dataset, x="x_1", y="x_2", color="Classification",
                 size=self.outl_scores, title='Outliers amd Inliers')
            return fig

        elif self.d == 3:
            # Construct pandas data for three dimensions
            dataset = pd.DataFrame({'x_1': x_all[:, 0], 'x_2': x_all[:, 1], 'x_3':x_all[:, 2],'Classification': labels})
            fig = px.scatter_3d(dataset, x="x_1", y="x_2", z="x_3",color="Classification",
                 size=self.outl_scores, title='Outliers amd Inliers')
            return fig
        else:
            # Use just two dimensions
            dataset = pd.DataFrame({'x_1': x_all[:, 0], 'x_2': x_all[:, 1], 'Classification': labels})
            fig = px.scatter(dataset, x="x_1", y="x_2", color="Classification",
                 size=self.outl_scores, title='Outliers amd Inliers')
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
        self.X = self.pca.transform(self.X)
        variance_components = self.pca.explained_variance_ratio_
        self.remained_variance = np.sum(variance_components[:m]) * 100

        # Update n,d
        self.n, self.d = self.X.shape

    def apply_LLE(self, m, neighbours=None, neigh_algo=None):
        ''' Perform LLE Locally Linear Embedding¶  on X and reduce to an m dimensional subspace
        Definition:  apply_LLE(X, m)
        Input:       X                  - NxD array of N data points with D features
                     m                  - int, dimension of the subspace to project
        '''
        embedding = LocallyLinearEmbedding(n_components=m, n_neighbors=neighbours, neighbors_algorithm='kd_tree')
        # Update X
        self.X = embedding.fit_transform(self.X)
        # Update n,d
        self.n, self.d = self.X.shape

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
                             options=get_files_dict(),
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
                                 {"label": "TSN", "value": 'tsn'}],
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
    [Output(component_id='outliers_scaled', component_property='figure')],
    [Input(component_id='eva_object', component_property='data'),
     dash.dependencies.Input('k_input', 'value'),
     Input(component_id='outlier detection method', component_property='value'),
     dash.dependencies.Input('threshold_input', 'value')]
)
def update_graph_dimred(selected_data_set, k_value,selected_dim_red_method, threshold):
    # Load the data from a pandas dataframe to a numpy array
    if selected_data_set is '':
        return dash.no_update
    elif selected_dim_red_method is '':
        return dash.no_update
    else:
        # Create objects again
        eva_data = EvaData()
        eva_data.load_data(file_name=selected_data_set)
        if selected_dim_red_method == 'gamma_index':
            if threshold == 0:
                fig = eva_data.gamma_index(k=k_value, scaling_factor=1)
                return [fig]
            else:
                f = eva_data.gamma_index(k=k_value, scaling_factor=1)
                fig = eva_data.visualize_outliers(threshold=threshold)
                return [fig]
        elif selected_dim_red_method == 'eps_ball':
            if threshold == 0:
                fig = eva_data.k_ball_index(k=k_value, scaling_factor=1)
                return [fig]
            else:
                f = eva_data.k_ball_index(k=k_value, scaling_factor=1)
                fig = eva_data.visualize_outliers(threshold=threshold)
                return [fig]
        else:
            return dash.no_update

@app.callback(
    dash.dependencies.Output('k_value', 'children'),
    [dash.dependencies.Input('k_input', 'value')])

def update_output(value):
    pass

@app.callback(
    [Output(component_id='red_dim_plot', component_property='figure'),
     Output(component_id='selected_reddim_method', component_property='data')],
    [Input(component_id='select red method', component_property='value'),
     Input(component_id='eva_object', component_property='data'),
     dash.dependencies.Input('ndim_input', 'value'),
     dash.dependencies.Input('lle_neighbours', 'value')]
)

def update_graph_reddim(selected_redim_method, selected_dataset, m, lle_neighbours):
    if selected_redim_method is not '':
        # Init class object
        eva_data = EvaData()
        eva_data.load_data(file_name=selected_dataset)
        # Apply method
        if selected_redim_method == 'pca':
            eva_data.apply_PCA(m=m)

        elif selected_redim_method == 'lle':
            eva_data.apply_LLE(m=m, neighbours=lle_neighbours)
        else:
            pass
        # Return scatter figure
        fig = eva_data.visualize_original_data()

        return fig, selected_redim_method
    else:
        return dash.no_update

@app.callback(
    [Output(component_id='outliers_red_dim', component_property='figure')],
    [Input(component_id='selected_reddim_method', component_property='data'),
     Input(component_id='eva_object', component_property='data'),
     dash.dependencies.Input('ndim_input', 'value'),
     dash.dependencies.Input('k_input_rdim', 'value'),
     Input(component_id='select outlier method reduced', component_property='value'),
     dash.dependencies.Input('threshold_input_rdim', 'value'),
     dash.dependencies.Input('lle_neighbours', 'value')]
)

def update_graph_reddim_outliers(selected_redim_method, selected_dataset, m, k, selected_outlier_method, threshold, lle_neighbours):
    if selected_redim_method is not '':
        if selected_outlier_method is not '':
            # Init class object
            eva_data = EvaData()
            eva_data.load_data(file_name=selected_dataset)
            # Apply method
            if selected_redim_method == 'pca':
                eva_data.apply_PCA(m=m)

            elif selected_redim_method == 'lle':
                eva_data.apply_LLE(m=m, neighbours=lle_neighbours)
            else:
                pass
            # Compute outlier score
            if selected_outlier_method == 'gamma_index':
                if threshold == 0:
                    fig = eva_data.gamma_index(k=k, scaling_factor=50)
                    return [fig]
                else:
                    f = eva_data.gamma_index(k=k, scaling_factor=50)
                    fig = eva_data.visualize_outliers(threshold=threshold)
                    return [fig]
            elif selected_outlier_method == 'eps_ball':
                if threshold == 0:
                    fig = eva_data.k_ball_index(k=k, scaling_factor=50)
                    return [fig]
                else:
                    f = eva_data.k_ball_index(k=k, scaling_factor=50)
                    fig = eva_data.visualize_outliers(threshold=threshold)
                    return [fig]
        else:
            return dash.no_update
    else:
        return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)

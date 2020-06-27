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
import pandas as pd
from visualisation_methods import VisualizationPlotly
from statistics_methods import DataStatistics


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
        dim_red_method = data_dict['algorithms']

        stats = DataStatistics()
        stats.load_data(file_name=data_file_name)
        stats.apply_pca(m=2)

        visualization_fest = VisualizationPlotly(stats.reduced_pandas_dataframe)


        fig_percentage_info = visualization_fest.pie_plot_percentages()


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
                        value=stats.n,
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

                ], className='one column', style={"margin-right": "27%"}),



                html.Div([
                    dcc.Dropdown(
                        id='histogram-dropdown',
                        placeholder="Select a feature",
                        options=[{'label': name, 'value': name} for name in list(stats.features)],
                        value= ''
                    ),
                ], style={'width': '15%'},className='three columns'),


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
                        value=stats.d,
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
                ], className='five columns'
                ),
                html.Div([
                    dcc.Graph(id='neighbours_plot', figure={}),
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
                    max=stats.d,
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
             dash.dependencies.Input('neighbours_input', 'value'),
             Input('histogram-dropdown', 'value')]
        )
        def update_graph(m, outl_display_option, box_dim, k_neighbours, histogram_feature):
            # Statistical Calculation load data
            # Apply PCA to the data
            stats.apply_pca(m=m)
            # Create fig to display pca data
            visualization = VisualizationPlotly(stats.reduced_pandas_dataframe)

            if outl_display_option == ['yes']:
                outl_pd = stats.reduced_pandas_dataframe[stats.reduced_pandas_dataframe.Classification.eq('Outlier')]
                fig = VisualizationPlotly(outl_pd).plot_data()
            else:
                fig = visualization.plot_data()

            # Box plot for the input dimension
            box_fig = visualization.box_plot_classifications(dim=box_dim)

            # K nearest neighbours distance histogram on the reduced/lower dimensional dataset
            if histogram_feature is not '':
                visual_original = VisualizationPlotly(stats.pa)
                kn_histogram_fig = visual_original.histogram_data(feature=histogram_feature)
            else:
                kn_histogram_fig = []
            #
            stats.graph_neighbours(n_neighbours=k_neighbours)
            graph_figure = visualization.graph_neighbours(edges=stats.edges, nodes=stats.nodes)
            return [fig, stats.remained_variance, box_fig, kn_histogram_fig, graph_figure]

        return self.dash_app.server


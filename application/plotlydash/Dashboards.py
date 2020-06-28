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



class DashBoard_DIMRED():

    def __init__(self, method, pandas_df_reduced_data):
        self.method = method
        self.pandas_df_reduced_data = pandas_df_reduced_data

    def build_dashboard(self):
        # Reduce data scatter plot
        vis = VisualizationPlotly(pd_data_frame=self.pandas_df_reduced_data)

        scatter_fig = vis.plot_data()

def dash_board_part(method, pandas_df_reduced_data):
    vis = VisualizationPlotly(pd_data_frame=pandas_df_reduced_data)

    scatter_fig = vis.plot_data()
    box_fig = vis.box_plot_classifications()
    dashboard = 1

    dashboard = html.Div([
                      html.Div([
                         dcc.Graph(id='reduced_data_plot_'+str(method), figure=scatter_fig)
                    ], className='five columns'
                    ),
                      html.Div([
                        dcc.Graph(id='box_outliers_plot_'+str(method), figure=box_fig)
                     ], className='five columns'
                     )
                   ], className="row"
               )


    return dashboard

class IrisDashboard(RemoteCSVDashboard):
    def __init__(self, server, stylesheets=external_stylesheets, prefix='/dashboard/', location=iris_config['location']):
        super().__init__(server, stylesheets, prefix, location)

        self.dash_app = dash.Dash(__name__, server=self.server,
                                  routes_pathname_prefix=self.prefix,
                                  external_stylesheets=self.stylesheets)

    def create_dashboard(self, data_dict):

        data_file_name = data_dict['location']# '/Users/albertorodriguez/Desktop/Current Courses/InternenServLab/EVA-merge_flask_dash/application/data/fishbowl_outl.csv'
        dim_red_method = data_dict['algorithms']

        #----------------- Build dashboard -----------------
        # Init list containing the dashboards
        dashboards_fig = []
        # Compute statistical methods, PCA,TSNE etc. according to the user's choice

        main_stats = DataStatistics()
        main_stats.load_data(file_name=data_file_name)

        # PCA , LLE etc initially computed for 2 dimensions, we may change that
        if dim_red_method['PCA']:
            main_stats.apply_pca()
            dashboards_fig.append(dash_board_part(method='pca',
                                                  pandas_df_reduced_data=main_stats.reduced_pandas_dataframe_pca))

        if dim_red_method['TSNE']:
            main_stats.apply_tsne()
            dashboards_fig.append(
                dash_board_part(method='tsne', pandas_df_reduced_data=main_stats.reduced_pandas_dataframe_tsne))

        if dim_red_method['LLE']:
            main_stats.apply_lle()

        #

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



            ], className='twelve columns offset-by-one')
        )


        return self.dash_app.server


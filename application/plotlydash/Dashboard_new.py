import flask
import dash_html_components as html
from abc import abstractmethod, ABC
from .assets.layout import html_layout
import dash
from config import iris_config, external_stylesheets
from application.plotlydash.dim_red_dshboards import DimRedDash
from statistics_methods import DataStatistics

from typing import List, Dict


class Dashboard(ABC):
    def __init__(self, server: flask.Flask, stylesheets: List[str], prefix: str = '/dash'):
        self.server = server
        self.stylesheets = stylesheets
        self.prefix = prefix
        self.dash_app = None

    @abstractmethod
    def create_dashboard(self, data_dict: Dict[str, str]):
        pass

    @abstractmethod
    def init_callbacks(self, target_column):
        pass

class RemoteCSVDashboard(Dashboard):
    def __init__(self, server, stylesheets, prefix, location):
        super().__init__(server, stylesheets, prefix)
        self.location = location

    def create_dashboard(self, target_column):
        pass

    def init_callbacks(self, target_column):
        pass

class FileDashboard(RemoteCSVDashboard):
    def __init__(self, server, stylesheets=external_stylesheets, prefix='/dashboard/', location=iris_config['location']):
        super().__init__(server, stylesheets, prefix, location)

        self.dash_app = dash.Dash(__name__, server=self.server,
                                  routes_pathname_prefix=self.prefix,
                                  external_stylesheets=self.stylesheets)

    def create_dashboard(self, data_dict: Dict[str, str]):
        data_file_name = data_dict['location']     # This is now in 'data' not 'application/data'
        #dim_red_methods = data_dict['algorithms']  # This is now a list.
        main_stats = DataStatistics()
        main_stats.load_data(data_file_name)

        # Init List containing all html div(...) dashboards
        dashboards_merged = []
        # Add title
        dashboards_merged.append(html.Div(
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
        ),)

        # Apply selected algorithms
        if 'PCA' in data_dict:
            main_stats.apply_pca()
            dashboard = DimRedDash(stats=main_stats, method='PCA', plot_options=data_dict['PCA'])

            dashboards_merged.append(dashboard.title)
            dashboards_merged.append(dashboard.dropdowns)
            dashboards_merged.append(dashboard.graph)

        if 'TSNE' in data_dict:
            main_stats.apply_tsne()
            dashboard = DimRedDash(stats=main_stats, method='TSNE', plot_options=data_dict['TSNE'])

            dashboards_merged.append(dashboard.title)
            dashboards_merged.append(dashboard.dropdowns)
            dashboards_merged.append(dashboard.graph)

        if 'LLE' in data_dict:
           main_stats.apply_lle()
           dashboard = DimRedDash(stats=main_stats, method='LLE', plot_options=data_dict['LLE'])

           dashboards_merged.append(dashboard.title)
           dashboards_merged.append(dashboard.dropdowns)
           dashboards_merged.append(dashboard.graph)

        # Merge
        print(type(dashboards_merged))
        print(len(dashboards_merged))
        # Merge all dashboards here

        self.dash_app.css.config.serve_locally = False
        # Boostrap CSS.
        self.dash_app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})  # noqa: E501
        self.dash_app.index_string = html_layout

        self.dash_app.layout = html.Div(
            html.Div( children= dashboards_merged, className='twelve columns offset-by-one')
        )

        return self.dash_app.server



#dashboard_config = {'location': session['filename'],
#                    'target': alg_form.target.data,
#                    'PCA': [],
#'LLE':['scatter,'box','kn']
 #                   }




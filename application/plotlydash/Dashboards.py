import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
from config import iris_config, external_stylesheets
from abc import abstractmethod, ABC


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
    def __init__(self, server, stylesheets=external_stylesheets, prefix='/iris_example/', location=iris_config['location']):
        super().__init__(server, stylesheets, prefix, location)

        self.dash_app = dash.Dash(__name__,server=self.server,
                                  routes_pathname_prefix=self.prefix,
                                  external_stylesheets=self.stylesheets)

    def init_callbacks(self, target_column):
        @self.dash_app.callback(
            Output('iris_graph', 'figure'),
            [Input('feat_1', 'value'),
             Input('feat_2', 'value')]
        )
        def update_graph(feat_1, feat_2):
            return {'data': [
                dict(
                    x=self.df[self.df[target_column] == name][feat_1],
                    y=self.df[self.df[target_column] == name][feat_2],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=name
                ) for name in self.df[target_column].unique()
            ],
                'layout': dict(
                    xaxis={'type': 'Linear', 'title': feat_1},
                    yaxis={'title': feat_2},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest',
                )}

    def create_dashboard(self, target_column=iris_config['target']):
        features = iris_config['features']  # Hardcoded


        self.dash_app.layout = html.Div( children = [
            html.Div([
                html.H2(" IRIS Dataset - Anamoly Detection"),
            ], className = "banner"),
            html.Div([iris_feature_selector(features[0], features, 'feat_1')],
                     style={'width': '48%', 'display': 'inline-block'}),
            html.Div([iris_feature_selector(features[1], features, 'feat_2')],
                     style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            dcc.Graph(id='iris_graph')
        ])
        self.init_callbacks(target_column)
        return self.dash_app.server
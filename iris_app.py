import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv')
features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']


def iris_feature_selector(default_value, component_id):
    return dcc.Dropdown(
            id=component_id,
            options=[{'label': feat, 'value': feat} for feat in features],
            value=default_value  # default value
            )


app.layout = html.Div([
    html.H1('IRIS Dataset'),
    html.Div([iris_feature_selector(features[0], 'feat_1')],
             style={'width': '48%', 'display': 'inline-block'}),
    html.Div([iris_feature_selector(features[1], 'feat_2')],
             style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    dcc.Graph(id='iris_graph')
])


@app.callback(
    Output('iris_graph', 'figure'),
    [Input('feat_1', 'value'),
     Input('feat_2', 'value')]
)
def update_graph(feat_1, feat_2):
    return {'data': [
                dict(
                    x=df[df['Name'] == name][feat_1],
                    y=df[df['Name'] == name][feat_2],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=name
                ) for name in df.Name.unique()
            ],
            'layout': dict(
                xaxis={'type': 'Linear', 'title': feat_1},
                yaxis={'title': feat_2},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest',
            )}


if __name__ == '__main__':
    app.run_server(debug=True)

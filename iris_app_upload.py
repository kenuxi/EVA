import base64
import io
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div( children = [
            html.Div([
                html.H2("Anamoly Detection"),
            ], className = "banner"),

    html.Div([
        html.Div([
                html.H3([
                    dcc.Dropdown(
                        id='Dropdown',
                        placeholder="Name"),
                ]),

            ], className="filter"),

            html.Div([
                html.H3([
                    dcc.Dropdown(
                        id='Dropdown1',
                        placeholder="Name"),
                ]),

            ], className="filter1") ,

    ]),

    html.Div([
        dcc.Upload(
            id='datatable-upload',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),

            style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                   'borderWidth': '1px', 'borderStyle': 'dashed',
                   'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                   },
        ),

        dash_table.DataTable(id='datatable-upload-container'),
        dcc.Graph(id='datatable-upload-graph')
    ]),

    html.Div(id='dd-output-container'),

])


def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    df_uploaded = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    return df_uploaded


@app.callback(Output('datatable-upload-container', 'data'),
              [Input('datatable-upload', 'contents')])
def update_output(contents):
    if contents is None:
        return [{}]
    df = parse_contents(contents)

    return df.to_dict('records')

@app.callback(Output('datatable-upload-graph', 'figure'),
[Input('datatable-upload-container', 'data'),
 Input('Dropdown', 'value'),
 Input('Dropdown1', 'value')])

def display_graph(rows, selector, selector1):
    df = pd.DataFrame(rows)

    print(df)

    return {'data': [
        dict(
            x=df[df.iloc[:,-1]  == name][selector],
            y=df[df.iloc[:,-1] == name][selector1],
            mode='markers',
            opacity=0.7,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=name
        ) for name in df.iloc[:,-1].unique()
    ],
        'layout': dict(
            xaxis={'type': 'Linear', 'title': selector},
            yaxis={'title': selector1},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest',
        )}

@app.callback(Output('Dropdown', 'options'),
              [Input('datatable-upload', 'contents')])


def parse_uploads(content):
    df = parse_contents(content)
    return [{'label': i, 'value': i } for i in df.columns[1:-1]]



@app.callback(Output('Dropdown1', 'options'),
              [Input('datatable-upload', 'contents')])


def parse_uploads1(content):
    df = parse_contents(content)
    return [{'label': i, 'value': i } for i in df.columns[1:-1]]

if __name__ == '__main__':
    app.run_server(debug=True)





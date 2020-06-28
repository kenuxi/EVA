import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from visualisation_methods import VisualizationPlotly
from statistics_methods import DataStatistics

class DimRedDash():
    def __init__(self, stats, method):
        '''

        Definition:  load_data(self, file_name)

        Input:       stats   - object, from DataStatistics class, containing all needed info/data for the plots
                     method  - str, 'PCA', 'LLE', 'TSNE' or 'KERNEL_PCA' indicating the method

        '''

        self.stats = stats
        self.method = method

        self.graph = self._get_graph()
        self.dropdowns = self._getdropdowns()
        self.title = self._gettitle()

    def _get_graph(self):
        ''' dashboard with the graphs plots
        '''

        # PCA CASE HERE, right now just 2 plots (scatter and boxplot)
        if self.method == 'PCA':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_pca)
            scatter_fig = visualisation.plot_data()
            box_fig = visualisation.box_plot_classifications()

            dashboard = html.Div([
                html.Div([
                    dcc.Graph(id='reduced_data_plot_pca', figure=scatter_fig)
                ], className='five columns'
                ),
                html.Div([
                    dcc.Graph(id='box_outliers_plot_pca', figure=box_fig)
                ], className='five columns'
                )
            ], className="row"

            )

        # LLE CASE HERE, right now just 2 plots (scatter and boxplot)
        if self.method == 'LLE':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_lle)
            scatter_fig_lle = visualisation.plot_data()
            box_fig_lle = visualisation.box_plot_classifications()

            dashboard = html.Div([
                html.Div([
                    dcc.Graph(id='reduced_data_plot_lle', figure=scatter_fig_lle)
                ], className='five columns'
                ),
                html.Div([
                    dcc.Graph(id='box_outliers_plot_lle', figure=box_fig_lle)
                ], className='five columns'
                )
            ], className="row"

            )


        # TSNE CASE  HERE, right now just 2 plots (scatter and boxplot)
        if self.method == 'TSNE':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_tsne)
            scatter_fig_tsne = visualisation.plot_data()
            box_fig_tsne = visualisation.box_plot_classifications()

            dashboard = html.Div([
                html.Div([
                    dcc.Graph(id='reduced_data_plot_tsne', figure=scatter_fig_tsne)
                ], className='five columns'
                ),
                html.Div([
                    dcc.Graph(id='box_outliers_plot_tsne', figure=box_fig_tsne)
                ], className='five columns'
                )
            ], className="row"

            )


        return dashboard

    def _getdropdowns(self):
        ''' dashboard dropdowns
        '''
        # PCA DROPDOWNS HERE
        if self.method == 'PCA':
            dashboard =  html.Div(
                [

                    html.Div([
                        daq.NumericInput(
                            id='rem_vari',
                            disabled=True,
                            min=0,
                            max=100,
                            size=120,
                            label='Remained Variance %',
                            labelPosition='bottom',
                            value=self.stats.remained_variance)
                    ],  className='two columns'),

                    html.Div([
                        daq.NumericInput(
                            id='red_dim_input',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ],className='two columns'),

                    html.Div([
                        daq.NumericInput(
                            id='box_red_dim',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='Boxplot dimension',
                            labelPosition='bottom',
                            value=2)
                    ], className='two columns'),

                ], className="row"
            )




        # LLE DROPDOWNS HERE
        if self.method == 'LLE':
            dashboard = []


        # TSNE DROPDOWNS HERE
        if self.method == 'TSNE':
            dashboard = []

        return dashboard

    def _gettitle(self):

        dashboard_title = html.Div(
            [
                html.H1(children=self.method,
                        className='nine columns',
                        style={
                            'color': '#111',
                            'font-family': 'sans-serif',
                            'font-size': 30,
                            'font-weight': 200,
                            'line-height': 58,
                             'margin': 0,
                             #'backgroundColor': '#DCDCDC'
                              },
                        ),
            ], className="row"
        )

        return dashboard_title



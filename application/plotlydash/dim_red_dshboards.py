import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from visualisation_methods import VisualizationPlotly


class DimRedDash():
    def __init__(self, stats, method):
        '''

        Definition:  load_data(self, file_name)

        Input:       stats   - object, from DataStatistics class, containing all needed info/data for the plots
                     method  - str, 'PCA', 'LLE', 'TSNE' , 'ISOMAP', 'UMAP', 'KERNEL_PCA' indicating the method

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
            scatter_fig_density = visualisation.plot_data_density()
            box_fig = visualisation.box_plot_classifications()

            # if pca_graph option
            self.stats.graph_neighbours(n_neighbours=4, algorithm='pca') # this should be done somewhere else
            pca_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)

            dashboard = html.Div([
                html.Div([
                    dcc.Graph(id='reduced_data_plot_pca', figure=scatter_fig)
                ], className='five columns'
                ),

                html.Div([
                    dcc.Graph(id='reduced_data_plot_density_pca', figure=scatter_fig_density)
                ], className='five columns'
                ),


                html.Div([
                    dcc.Graph(id='box_outliers_plot_pca', figure=box_fig)
                ], className='five columns'
                ),
                html.Div([
                    dcc.Graph(id='connected_graph_figure_pca', figure=pca_graph),
                ], className='five columns'
                ),

            ], className="row")


        # LLE CASE HERE, right now just 2 plots (scatter and boxplot)
        if self.method == 'LLE':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_lle)
            scatter_fig_lle = visualisation.plot_data()
            box_fig_lle = visualisation.box_plot_classifications()
            scatter_fig_density_lle = visualisation.plot_data_density()

            # if pca_graph option
            self.stats.graph_neighbours(n_neighbours=4, algorithm='lle') # this should be done somewhere else
            lle_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)

            dashboard = html.Div([
                html.Div([
                    dcc.Graph(id='reduced_data_plot_lle', figure=scatter_fig_lle)
                ], className='five columns'
                ),

                html.Div([
                    dcc.Graph(id='reduced_data_plot_density_lle', figure=scatter_fig_density_lle)
                ], className='five columns'
                ),

                html.Div([
                    dcc.Graph(id='box_outliers_plot_lle', figure=box_fig_lle)
                ], className='five columns'
                ),
                html.Div([
                    dcc.Graph(id='connected_graph_figure_lle', figure=lle_graph),
                ], className='five columns'
                )
            ], className="row"

            )


        # TSNE CASE  HERE, right now just 2 plots (scatter and boxplot)
        if self.method == 'TSNE':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_tsne)
            scatter_fig_tsne = visualisation.plot_data()
            box_fig_tsne = visualisation.box_plot_classifications()

            # if pca_graph option
            self.stats.graph_neighbours(n_neighbours=4, algorithm='tsne') # this should be done somewhere else
            tsne_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)

            tsne_board = []

            # Build subboard according to the user specifications # still to be implemented
            tsne_board.append(html.Div([
                dcc.Graph(id='reduced_data_plot_tsne', figure=scatter_fig_tsne)
            ], className='five columns'
            ),)

            tsne_board.append(html.Div([
                dcc.Graph(id='box_outliers_plot_tsne', figure=box_fig_tsne)
            ], className='five columns'
            ),)

            tsne_board.append(html.Div([
                dcc.Graph(id='connected_graph_figure_tsne', figure=tsne_graph)
            ], className='five columns'
            ),)
            # Init List corresponding to the PCA Dashboardb PLOTS
            dashboard = html.Div(children=tsne_board, className="row")

        # ISOMAP CASE HERE, right now just 2 plots (scatter and boxplot)
        if self.method == 'ISOMAP':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_isomap)
            scatter_fig_isomap = visualisation.plot_data()
            box_fig_isomap = visualisation.box_plot_classifications()
            scatter_fig_density_isomap = visualisation.plot_data_density()

            # if pca_graph option
            self.stats.graph_neighbours(n_neighbours=6, algorithm='isomap')  # this should be done somewhere else
            isomap_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)

            dashboard = html.Div([
                html.Div([
                    dcc.Graph(id='reduced_data_plot_isomap', figure=scatter_fig_isomap)
                ], className='five columns'
                ),

                html.Div([
                    dcc.Graph(id='reduced_data_plot_density_isomap', figure=scatter_fig_density_isomap)
                ], className='five columns'
                ),

                html.Div([
                    dcc.Graph(id='box_outliers_plot_isomap', figure=box_fig_isomap)
                ], className='five columns'
                ),
                html.Div([
                    dcc.Graph(id='connected_graph_figure_isomap', figure=isomap_graph),
                ], className='five columns'
                )
            ], className="row"

            )

        # UMAP CASE HERE, right now just 2 plots (scatter and boxplot)
        if self.method == 'UMAP':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_umap)
            scatter_fig_umap = visualisation.plot_data()
            box_fig_umap = visualisation.box_plot_classifications()
            scatter_fig_density_umap = visualisation.plot_data_density()

            # if pca_graph option
            self.stats.graph_neighbours(n_neighbours=6, algorithm='umap')  # this should be done somewhere else
            umap_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)

            dashboard = html.Div([
                html.Div([
                    dcc.Graph(id='reduced_data_plot_umap', figure=scatter_fig_umap)
                ], className='five columns'
                ),

                html.Div([
                    dcc.Graph(id='reduced_data_plot_density_umap', figure=scatter_fig_density_umap)
                ], className='five columns'
                ),

                html.Div([
                    dcc.Graph(id='box_outliers_plot_umap', figure=box_fig_umap)
                ], className='five columns'
                ),
                html.Div([
                    dcc.Graph(id='connected_graph_figure_umap', figure=umap_graph),
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
                        dcc.Checklist(
                            id='outlier_only_options_pca',
                            options=[
                                {'label': 'Only show Outliers', 'value': 'yes'}
                            ],
                        ),
                    ], className='one column'),

                    html.Div([
                        daq.NumericInput(
                            id='red_dim_input_pca',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ],className='two columns'),

                    html.Div([
                        daq.NumericInput(
                            id='box_red_dim_pca',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='Boxplot dimension',
                            labelPosition='bottom',
                            value=2)
                    ], className='two columns'),

                ], className="row"
            )

        # TSNE OPTIONS HERE
        if self.method == 'TSNE':
            dashboard = html.Div(
                [
                    html.Div([
                        daq.NumericInput(
                            id='red_dim_input_tsne',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ],className='two columns'),

                    html.Div([
                        dcc.Checklist(
                            id='outlier_only_options_tsne',
                            options=[
                                {'label': 'Only show Outliers', 'value': 'yes'}
                            ],
                        ),
                    ], className='one column'),

                    html.Div([
                        daq.NumericInput(
                            id='perplexity_tsne',
                            min=1,
                            max=100,
                            size=120,
                            label='Perplexity',
                            labelPosition='bottom',
                            value=30),
                    ], className='two columns'),

                    html.Div([
                        daq.NumericInput(
                            id='box_red_dim_tsne',
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
            dashboard = html.Div(
                [
                    html.Div([
                        daq.NumericInput(
                            id='red_dim_input_lle',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ],className='two columns'),

                    html.Div([
                        dcc.Checklist(
                            id='outlier_only_options_lle',
                            options=[
                                {'label': 'Only show Outliers', 'value': 'yes'}
                            ],
                        ),
                    ], className='one column'),

                    html.Div([
                        daq.NumericInput(
                            id='nieghbours_lle',
                            min=1,
                            max=self.stats.n-1,
                            size=120,
                            label='K-Neighbours',
                            labelPosition='bottom',
                            value=5),
                    ], className='two columns'),

                    html.Div([
                        daq.NumericInput(
                            id='box_red_dim_lle',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='Boxplot dimension',
                            labelPosition='bottom',
                            value=2)
                    ], className='two columns'),

                ], className="row"
            )

        # ISOMAP DROPDOWNS HERE
        if self.method == 'ISOMAP':
            dashboard = html.Div(
                [
                    html.Div([
                        daq.NumericInput(
                            id='red_dim_input_isomap',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ], className='two columns'),

                    html.Div([
                        dcc.Checklist(
                            id='outlier_only_options_isomap',
                            options=[
                                {'label': 'Only show Outliers', 'value': 'yes'}
                            ],
                        ),
                    ], className='one column'),

                    html.Div([
                        daq.NumericInput(
                            id='nieghbours_isomap',
                            min=1,
                            max=self.stats.n - 1,
                            size=120,
                            label='K-Neighbours',
                            labelPosition='bottom',
                            value=6),
                    ], className='two columns'),

                    html.Div([
                        daq.NumericInput(
                            id='box_red_dim_isomap',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='Boxplot dimension',
                            labelPosition='bottom',
                            value=2)
                    ], className='two columns'),

                ], className="row"
            )

        # UMAP DROPDOWNS HERE
        if self.method == 'UMAP':
            dashboard = html.Div(
                [
                    html.Div([
                        daq.NumericInput(
                            id='red_dim_input_umap',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ], className='two columns'),

                    html.Div([
                        dcc.Checklist(
                            id='outlier_only_options_umap',
                            options=[
                                {'label': 'Only show Outliers', 'value': 'yes'}
                            ],
                        ),
                    ], className='one column'),

                    html.Div([
                        daq.NumericInput(
                            id='nieghbours_umap',
                            min=1,
                            max=self.stats.n - 1,
                            size=120,
                            label='K-Neighbours',
                            labelPosition='bottom',
                            value=6),
                    ], className='two columns'),

                    html.Div([
                        daq.NumericInput(
                            id='box_red_dim_umap',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='Boxplot dimension',
                            labelPosition='bottom',
                            value=2)
                    ], className='two columns'),

                ], className="row"
            )

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



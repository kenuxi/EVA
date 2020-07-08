import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from visualisation_methods import VisualizationPlotly


class DimRedDash():
    def __init__(self, stats, method, plot_options):
        '''

        Definition:  load_data(self, file_name)

        Input:       stats   - object, from DataStatistics class, containing all needed info/data for the plots
                     method  - str, 'PCA', 'LLE', 'TSNE' or 'KERNEL_PCA' indicating the method

        '''

        self.stats = stats
        self.method = method
        self.plot_options = plot_options

        self.graph = self._get_graph()
        self.dropdowns = self._getdropdowns()
        self.callbacks = self._getcallbacks()
        self.title = self._gettitle()

    def _get_graph(self):
        ''' dashboard with the graphs plots
        '''

        # PCA CASE HERE, right now just 2 plots (scatter and boxplot)
        if self.method == 'PCA':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_pca)

            # Depending on the plot_options the user selected, produce corresponding plots and append them to the
            # list
            pca_plots = []
            if 'scatter' in self.plot_options:
                scatter_fig = visualisation.plot_data()
                pca_plots.append(html.Div([
                    dcc.Graph(id='reduced_data_plot_pca', figure=scatter_fig)
                ], className='five columns'
                ))

            if 'box' in self.plot_options:
                box_fig = visualisation.box_plot_classifications()
                pca_plots.append(html.Div([
                    dcc.Graph(id='box_outliers_plot_pca', figure=box_fig)
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='pca')  # this should be done somewhere else
                pca_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                pca_plots.append(                html.Div([
                    dcc.Graph(id='connected_graph_figure_pca', figure=pca_graph),
                ], className='five columns'
                ))

            if 'dendogram' in self.plot_options:
                pca_dendo = visualisation.plot_dendrogram()
                pca_plots.append(html.Div([
                    dcc.Graph(id='dendogram_pca', figure=pca_dendo),
                ], className='five columns'
                ))
            dashboard = html.Div(children=pca_plots, className="row")

        if self.method == 'LLE':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_lle)

            # Depending on the plot_options the user selected, produce corresponding plots and append them to the
            # list
            lle_plots = []
            if 'scatter' in self.plot_options:
                scatter_fig_lle = visualisation.plot_data()
                lle_plots.append(html.Div([
                    dcc.Graph(id='reduced_data_plot_lle', figure=scatter_fig_lle)
                ], className='five columns'
                ))

            if 'box' in self.plot_options:
                box_fig_lle = visualisation.box_plot_classifications()
                lle_plots.append(html.Div([
                    dcc.Graph(id='box_outliers_plot_lle', figure=box_fig_lle)
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='lle')  # this should be done somewhere else
                lle_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                lle_plots.append(                html.Div([
                    dcc.Graph(id='connected_graph_figure_lle', figure=lle_graph),
                ], className='five columns'
                ))

            if 'dendogram' in self.plot_options:
                lle_dendo = visualisation.plot_dendrogram()
                lle_plots.append(html.Div([
                    dcc.Graph(id='dendogram_lle', figure=lle_dendo),
                ], className='five columns'
                ))

            dashboard = html.Div(children=lle_plots, className="row")

        if self.method == 'TSNE':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_tsne)

            # Depending on the plot_options the user selected, produce corresponding plots and append them to the
            # list
            tsne_plots = []
            if 'scatter' in self.plot_options:
                scatter_fig_tsne = visualisation.plot_data()
                tsne_plots.append(html.Div([
                    dcc.Graph(id='reduced_data_plot_tsne', figure=scatter_fig_tsne)
                ], className='five columns'
                ))

            if 'box' in self.plot_options:
                box_fig_tsne = visualisation.box_plot_classifications()
                tsne_plots.append(html.Div([
                    dcc.Graph(id='box_outliers_plot_tsne', figure=box_fig_tsne)
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='tsne')  # this should be done somewhere else
                tsne_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                tsne_plots.append(                html.Div([
                    dcc.Graph(id='connected_graph_figure_tsne', figure=tsne_graph),
                ], className='five columns'
                ))

            if 'dendogram' in self.plot_options:
                tsne_dendo = visualisation.plot_dendrogram()
                tsne_plots.append(html.Div([
                    dcc.Graph(id='dendogram_tsne', figure=tsne_dendo),
                ], className='five columns'
                ))

            dashboard = html.Div(children=tsne_plots, className="row")

        if self.method == 'UMAP':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_umap)

            # Depending on the plot_options the user selected, produce corresponding plots and append them to the
            # list
            umap_plots = []
            if 'scatter' in self.plot_options:
                scatter_fig = visualisation.plot_data()
                umap_plots.append(html.Div([
                    dcc.Graph(id='reduced_data_plot_umap', figure=scatter_fig)
                ], className='five columns'
                ))

            if 'box' in self.plot_options:
                box_fig = visualisation.box_plot_classifications()
                umap_plots.append(html.Div([
                    dcc.Graph(id='box_outliers_plot_umap', figure=box_fig)
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='umap')  # this should be done somewhere else
                umap_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                umap_plots.append(html.Div([
                    dcc.Graph(id='connected_graph_figure_umap', figure=umap_graph),
                ], className='five columns'
                ))

            if 'dendogram' in self.plot_options:
                umap_dendo = visualisation.plot_dendrogram()
                umap_plots.append(html.Div([
                    dcc.Graph(id='dendogram_umap', figure=umap_dendo),
                ], className='five columns'
                ))

            dashboard = html.Div(children=umap_plots, className="row")

        if self.method == 'ISOMAP':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_isomap)

            # Depending on the plot_options the user selected, produce corresponding plots and append them to the
            # list
            isomap_plots = []
            if 'scatter' in self.plot_options:
                scatter_fig = visualisation.plot_data()
                isomap_plots.append(html.Div([
                    dcc.Graph(id='reduced_data_plot_isomap', figure=scatter_fig)
                ], className='five columns'
                ))

            if 'box' in self.plot_options:
                box_fig = visualisation.box_plot_classifications()
                isomap_plots.append(html.Div([
                    dcc.Graph(id='box_outliers_plot_isomap', figure=box_fig)
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='isomap')  # this should be done somewhere else
                isomap_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                isomap_plots.append(html.Div([
                    dcc.Graph(id='connected_graph_figure_umap', figure=isomap_graph),
                ], className='five columns'
                ))

            if 'dendogram' in self.plot_options:
                isomap_dendo = visualisation.plot_dendrogram()
                isomap_plots.append(html.Div([
                    dcc.Graph(id='dendogram_isomap', figure=isomap_dendo),
                ], className='five columns'
                ))

            dashboard = html.Div(children=isomap_plots, className="row")

        return dashboard

    def _getdropdowns(self):
        ''' dashboard dropdowns
        '''
        # PCA DROPDOWNS HERE
        if self.method == 'PCA':
            pca_dropdowns = []
            if 'scatter' in self.plot_options:
                pca_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='rem_vari',
                            disabled=True,
                            min=0,
                            max=100,
                            size=120,
                            label='Remained Variance %',
                            labelPosition='bottom',
                            value=self.stats.remained_variance)
                    ],  className='two columns'))

                pca_dropdowns.append(html.Div([
                        dcc.Checklist(
                            id='outlier_only_options_pca',
                            options=[
                                {'label': 'Only show Outliers', 'value': 'yes'}
                            ],
                        ),
                    ], className='one column'))

                pca_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='red_dim_input_pca',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ], className='two columns'))

            if 'box' in self.plot_options:
                pca_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='box_red_dim_pca',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='Boxplot dimension',
                            labelPosition='bottom',
                            value=2)
                    ], className='two columns'))

            if 'graph' in self.plot_options:
                pass


            dashboard = html.Div(children=pca_dropdowns, className="row")

        # TSNE OPTIONS HERE
        if self.method == 'TSNE':
            tsne_dropdowns = []

            if 'scatter' in self.plot_options:
                tsne_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='red_dim_input_tsne',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ], className='two columns'))

                tsne_dropdowns.append(html.Div([
                        dcc.Checklist(
                            id='outlier_only_options_tsne',
                            options=[
                                {'label': 'Only show Outliers', 'value': 'yes'}
                            ],
                        ),
                    ], className='one column'))

                tsne_dropdowns.append(                    html.Div([
                        daq.NumericInput(
                            id='perplexity_tsne',
                            min=1,
                            max=100,
                            size=120,
                            label='Perplexity',
                            labelPosition='bottom',
                            value=30),
                    ], className='two columns'))

            if 'box' in self.plot_options:
                tsne_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='box_red_dim_tsne',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='Boxplot dimension',
                            labelPosition='bottom',
                            value=2)
                    ], className='two columns'))

            if 'graph' in self.plot_options:
                pass

            dashboard = html.Div(children=tsne_dropdowns, className="row")

        # LLE DROPDOWNS HERE
        if self.method == 'LLE':
            lle_dropdowns = []

            if 'scatter' in self.plot_options:
                lle_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='red_dim_input_lle',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ],className='two columns'))

                lle_dropdowns.append(html.Div([
                        dcc.Checklist(
                            id='outlier_only_options_lle',
                            options=[
                                {'label': 'Only show Outliers', 'value': 'yes'}
                            ],
                        ),
                    ], className='one column'))

                lle_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='nieghbours_lle',
                            min=1,
                            max=self.stats.n-1,
                            size=120,
                            label='K-Neighbours',
                            labelPosition='bottom',
                            value=5),
                    ], className='two columns'))

            if 'box' in self.plot_options:
                lle_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='box_red_dim_lle',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='Boxplot dimension',
                            labelPosition='bottom',
                            value=2)
                    ], className='two columns'))

            if 'graph' in self.plot_options:
                pass

            dashboard = html.Div(children=lle_dropdowns, className="row")


        # TSNE OPTIONS HERE
        if self.method == 'UMAP':
            umap_dropdowns = []

            if 'scatter' in self.plot_options:
                umap_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='red_dim_input_umap',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ], className='two columns'))

                umap_dropdowns.append(html.Div([
                        dcc.Checklist(
                            id='outlier_only_options_umap',
                            options=[
                                {'label': 'Only show Outliers', 'value': 'yes'}
                            ],
                        ),
                    ], className='one column'))

                umap_dropdowns.append(                    html.Div([
                        daq.NumericInput(
                            id='kneighbours_umap',
                            min=1,
                            max=100,
                            size=120,
                            label='k-Neighbours',
                            labelPosition='bottom',
                            value=30),
                    ], className='two columns'))

            if 'box' in self.plot_options:
                umap_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='box_red_dim_umap',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='Boxplot dimension',
                            labelPosition='bottom',
                            value=2)
                    ], className='two columns'))

            if 'graph' in self.plot_options:
                pass

            dashboard = html.Div(children=umap_dropdowns, className="row")

        # ISOMAP DROPDOWNS HERE
        if self.method == 'ISOMAP':
            isomap_dropdowns = []

            if 'scatter' in self.plot_options:
                isomap_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='red_dim_input_isomap',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ], className='two columns'))

                isomap_dropdowns.append(html.Div([
                        dcc.Checklist(
                            id='outlier_only_options_isomap',
                            options=[
                                {'label': 'Only show Outliers', 'value': 'yes'}
                            ],
                        ),
                    ], className='one column'))

                isomap_dropdowns.append(                    html.Div([
                        daq.NumericInput(
                            id='kneighbours_isomap',
                            min=1,
                            max=100,
                            size=120,
                            label='k-Neighbours',
                            labelPosition='bottom',
                            value=30),
                    ], className='two columns'))

            if 'box' in self.plot_options:
                isomap_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='box_red_dim_isomap',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='Boxplot dimension',
                            labelPosition='bottom',
                            value=2)
                    ], className='two columns'))

            if 'graph' in self.plot_options:
                pass

            dashboard = html.Div(children=isomap_dropdowns, className="row")

        return dashboard

    def _getcallbacks(self):
        pass
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



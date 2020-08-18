import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from eva.visualisation_methods import VisualizationPlotly


class DimRedDash():
    def __init__(self, stats, method, plot_options):
        '''

        Definition:  load_data(self, file_name)

        Input:       stats   - object, from DataStatistics class, containing all needed info/data for the plots
                     method  - str, 'PCA', 'LLE', 'TSNE', 'MDS' or 'KERNEL_PCA' indicating the method

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
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_pca,
                                                column_name=self.stats.label_column)

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

            if 'dendrogram' in self.plot_options:
                pca_dendo = visualisation.plot_dendrogram()
                pca_plots.append(html.Div([
                    dcc.Graph(id='dendrogram_pca', figure=pca_dendo),
                ], className='five columns'
                ))

            if 'density' in self.plot_options:
                pca_density = visualisation.plot_data_density()
                pca_plots.append(html.Div([
                    dcc.Graph(id='density_pca', figure=pca_density),
                ], className='five columns'
                ))

            if 'heat' in self.plot_options:
                pca_heat = visualisation.plot_data_heat()
                pca_plots.append(html.Div([
                    dcc.Graph(id='heat_pca', figure=pca_heat),
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='pca')  # this should be done somewhere else
                pca_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                pca_plots.append(html.Div([
                    dcc.Graph(id='connected_graph_figure_pca', figure=pca_graph),
                ], className='five columns'
                ))

            dashboard = html.Div(children=pca_plots, className="row")

        if self.method == 'LLE':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_lle,
                                                column_name=self.stats.label_column)

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

            if 'dendrogram' in self.plot_options:
                lle_dendo = visualisation.plot_dendrogram()
                lle_plots.append(html.Div([
                    dcc.Graph(id='dendrogram_lle', figure=lle_dendo),
                ], className='five columns'
                ))

            if 'density' in self.plot_options:
                lle_density = visualisation.plot_data_density()
                lle_plots.append(html.Div([
                    dcc.Graph(id='density_lle', figure=lle_density),
                ], className='five columns'
                ))

            if 'heat' in self.plot_options:
                lle_heat = visualisation.plot_data_heat()
                lle_plots.append(html.Div([
                    dcc.Graph(id='heat_lle', figure=lle_heat),
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='lle')  # this should be done somewhere else
                lle_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                lle_plots.append(                html.Div([
                    dcc.Graph(id='connected_graph_figure_lle', figure=lle_graph),
                ], className='five columns'
                ))

            dashboard = html.Div(children=lle_plots, className="row")

        if self.method == 'TSNE':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_tsne,
                                                column_name=self.stats.label_column)

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

            if 'dendrogram' in self.plot_options:
                tsne_dendo = visualisation.plot_dendrogram()
                tsne_plots.append(html.Div([
                    dcc.Graph(id='dendrogram_tsne', figure=tsne_dendo),
                ], className='five columns'
                ))

            if 'density' in self.plot_options:
                tsne_density = visualisation.plot_data_density()
                tsne_plots.append(html.Div([
                    dcc.Graph(id='density_tsne', figure=tsne_density),
                ], className='five columns'
                ))

            if 'heat' in self.plot_options:
                tsne_heat = visualisation.plot_data_heat()
                tsne_plots.append(html.Div([
                    dcc.Graph(id='heat_tsne', figure=tsne_heat),
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='tsne')  # this should be done somewhere else
                tsne_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                tsne_plots.append(                html.Div([
                    dcc.Graph(id='connected_graph_figure_tsne', figure=tsne_graph),
                ], className='five columns'
                ))

            dashboard = html.Div(children=tsne_plots, className="row")

        if self.method == 'UMAP':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_umap,
                                                column_name=self.stats.label_column)

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

            if 'dendrogram' in self.plot_options:
                umap_dendo = visualisation.plot_dendrogram()
                umap_plots.append(html.Div([
                    dcc.Graph(id='dendrogram_umap', figure=umap_dendo),
                ], className='five columns'
                ))

            if 'density' in self.plot_options:
                umap_density = visualisation.plot_data_density()
                umap_plots.append(html.Div([
                    dcc.Graph(id='density_umap', figure=umap_density),
                ], className='five columns'
                ))

            if 'heat' in self.plot_options:
                umap_heat = visualisation.plot_data_heat()
                umap_plots.append(html.Div([
                    dcc.Graph(id='heat_umap', figure=umap_heat),
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='umap')  # this should be done somewhere else
                umap_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                umap_plots.append(html.Div([
                    dcc.Graph(id='connected_graph_figure_umap', figure=umap_graph),
                ], className='five columns'
                ))

            dashboard = html.Div(children=umap_plots, className="row")

        if self.method == 'KMAP':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_kmap,
                                                column_name=self.stats.label_column)

            # Depending on the plot_options the user selected, produce corresponding plots and append them to the
            # list
            kmap_plots = []
            if 'scatter' in self.plot_options:
                scatter_fig = visualisation.plot_data()
                kmap_plots.append(html.Div([
                    dcc.Graph(id='reduced_data_plot_kmap', figure=scatter_fig)
                ], className='five columns'
                ))

            if 'box' in self.plot_options:
                box_fig = visualisation.box_plot_classifications()
                kmap_plots.append(html.Div([
                    dcc.Graph(id='box_outliers_plot_kmap', figure=box_fig)
                ], className='five columns'
                ))

            if 'dendrogram' in self.plot_options:
                kmap_dendo = visualisation.plot_dendrogram()
                kmap_plots.append(html.Div([
                    dcc.Graph(id='dendrogram_kmap', figure=kmap_dendo),
                ], className='five columns'
                ))

            if 'density' in self.plot_options:
                kmap_density = visualisation.plot_data_density()
                kmap_plots.append(html.Div([
                    dcc.Graph(id='density_kmap', figure=kmap_density),
                ], className='five columns'
                ))

            if 'heat' in self.plot_options:
                kmap_heat = visualisation.plot_data_heat()
                kmap_plots.append(html.Div([
                    dcc.Graph(id='heat_kmap', figure=kmap_heat),
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='kmap')  # this should be done somewhere else
                kmap_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                kmap_plots.append(html.Div([
                    dcc.Graph(id='connected_graph_figure_kmap', figure=kmap_graph),
                ], className='five columns'
                ))

            dashboard = html.Div(children=kmap_plots, className="row")

        if self.method == 'ISOMAP':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_isomap,
                                                column_name=self.stats.label_column)

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

            if 'dendrogram' in self.plot_options:
                isomap_dendo = visualisation.plot_dendrogram()
                isomap_plots.append(html.Div([
                    dcc.Graph(id='dendrogram_isomap', figure=isomap_dendo),
                ], className='five columns'
                ))

            if 'density' in self.plot_options:
                isomap_density = visualisation.plot_data_density()
                isomap_plots.append(html.Div([
                    dcc.Graph(id='density_isomap', figure=isomap_density),
                ], className='five columns'
                ))

            if 'heat' in self.plot_options:
                isomap_heat = visualisation.plot_data_heat()
                isomap_plots.append(html.Div([
                    dcc.Graph(id='heat_isomap', figure=isomap_heat),
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='isomap')  # this should be done somewhere else
                isomap_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                isomap_plots.append(html.Div([
                    dcc.Graph(id='connected_graph_figure_isomap', figure=isomap_graph),
                ], className='five columns'
                ))

            dashboard = html.Div(children=isomap_plots, className="row")

        if self.method == 'MDS':
            visualisation = VisualizationPlotly(pd_data_frame=self.stats.reduced_pandas_dataframe_mds,
                                                column_name=self.stats.label_column)

            # Depending on the plot_options the user selected, produce corresponding plots and append them to the
            # list
            mds_plots = []
            if 'scatter' in self.plot_options:
                scatter_fig = visualisation.plot_data()
                mds_plots.append(html.Div([
                    dcc.Graph(id='reduced_data_plot_mds', figure=scatter_fig)
                ], className='five columns'
                ))

            if 'box' in self.plot_options:
                box_fig = visualisation.box_plot_classifications()
                mds_plots.append(html.Div([
                    dcc.Graph(id='box_outliers_plot_mds', figure=box_fig)
                ], className='five columns'
                ))

            if 'dendrogram' in self.plot_options:
                mds_dendo = visualisation.plot_dendrogram()
                mds_plots.append(html.Div([
                    dcc.Graph(id='dendrogram_mds', figure=mds_dendo),
                ], className='five columns'
                ))

            if 'density' in self.plot_options:
                mds_density = visualisation.plot_data_density()
                mds_plots.append(html.Div([
                    dcc.Graph(id='density_mds', figure=mds_density),
                ], className='five columns'
                ))

            if 'heat' in self.plot_options:
                mds_heat = visualisation.plot_data_heat()
                mds_plots.append(html.Div([
                    dcc.Graph(id='heat_mds', figure=mds_heat),
                ], className='five columns'
                ))

            if 'k' in self.plot_options:
                self.stats.graph_neighbours(n_neighbours=4, algorithm='mds')  # this should be done somewhere else
                mds_graph = visualisation.graph_neighbours(self.stats.edges, self.stats.nodes)
                mds_plots.append(html.Div([
                    dcc.Graph(id='connected_graph_figure_mds', figure=mds_graph),
                ], className='five columns'
                ))

            dashboard = html.Div(children=mds_plots, className="row")

        return dashboard

    def _getdropdowns(self):
        ''' dashboard dropdowns
        '''

        # PCA DROPDOWNS HERE
        if self.method == 'PCA':
            print('I am here')
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
                        daq.NumericInput(
                            id='red_dim_input_pca',
                            min=1,
                            max=min([self.stats.d, 3]),
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2
                            ),
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
                            max=min([self.stats.d, 3]),
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ], className='two columns'))


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
                            max=min([self.stats.d, 3]),
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ],className='two columns'))


                lle_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='neighbours_lle',
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


        # UMAP OPTIONS HERE
        if self.method == 'UMAP':
            umap_dropdowns = []

            if 'scatter' in self.plot_options:
                umap_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='red_dim_input_umap',
                            min=1,
                            max=min([self.stats.d, 3]),
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2),
                    ], className='two columns'))


                umap_dropdowns.append(                    html.Div([
                        daq.NumericInput(
                            id='kneighbours_umap',
                            min=1,
                            max=100,
                            size=120,
                            label='k-Neighbours',
                            labelPosition='bottom',
                            value=5),
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

        # KMAP OPTIONS HERE
        if self.method == 'KMAP':
            kmap_dropdowns = []

            if 'scatter' in self.plot_options:
                kmap_dropdowns.append(html.Div([
                    daq.NumericInput(
                        id='red_dim_input_kmap',
                        min=1,
                        max=self.stats.d_red,
                        size=120,
                        label='subspace dimension',
                        labelPosition='bottom',
                        value=2),
                ], className='two columns'))



                kmap_dropdowns.append(html.Div([
                    daq.NumericInput(
                        id='kneighbours_kmap',
                        min=1,
                        max=100,
                        size=120,
                        label='k-Neighbours',
                        labelPosition='bottom',
                        value=30),
                ], className='two columns'))

                kmap_dropdowns.append(html.Div([
                    dcc.Dropdown(
                        id='algs_kmap',
                        options=[
                            {'label': 'PCA', 'value': 'PCA'},
                            {'label': 'UMAP', 'value': 'UMAP'},
                            {'label': 'LLE', 'value': 'LLE'},
                            {'label': 'TSNE', 'value': 'TSNE'},
                            {'label': 'ISOMAP', 'value': 'ISOMAP'},
                            {'label': 'MDS', 'value': 'MDS'}
                        ],
                        value='PCA'),
                ], className='two columns'))

            if 'box' in self.plot_options:
                kmap_dropdowns.append(html.Div([
                    daq.NumericInput(
                        id='box_red_dim_kmap',
                        min=1,
                        max=self.stats.d_red,
                        size=120,
                        label='Boxplot dimension',
                        labelPosition='bottom',
                        value=2)
                ], className='two columns'))

            if 'graph' in self.plot_options:
                pass

            dashboard = html.Div(children=kmap_dropdowns, className="row")

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

        # MDS DROPDOWNS HERE
        if self.method == 'MDS':
            mds_dropdowns = []
            if 'scatter' in self.plot_options:

                mds_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='red_dim_input_mds',
                            min=1,
                            max=min([self.stats.d, 3]),
                            size=120,
                            label='subspace dimension',
                            labelPosition='bottom',
                            value=2
                            ),
                    ], className='two columns'))

            if 'box' in self.plot_options:
                mds_dropdowns.append(html.Div([
                        daq.NumericInput(
                            id='box_red_dim_mds',
                            min=1,
                            max=self.stats.d_red,
                            size=120,
                            label='Boxplot dimension',
                            labelPosition='bottom',
                            value=2)
                    ], className='two columns'))

            if 'graph' in self.plot_options:
                pass

            dashboard = html.Div(children=mds_dropdowns, className="row")

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



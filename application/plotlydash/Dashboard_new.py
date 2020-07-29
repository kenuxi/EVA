import flask
import dash_html_components as html
from abc import abstractmethod, ABC
from .assets.layout import html_layout
import dash
from config import iris_config, external_stylesheets
from application.plotlydash.dim_red_dshboards import DimRedDash
from statistics_methods import DataStatistics
from dash.dependencies import Input, Output
from visualisation_methods import  VisualizationPlotly

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

    def create_dashboard(self, data_dict: Dict):
        main_stats = data_dict['ds']
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
        if data_dict['PCA']:
            main_stats.apply_pca()
            dashboard = DimRedDash(stats=main_stats, method='PCA', plot_options=data_dict['PCA'])

            dashboards_merged.append(dashboard.title)
            dashboards_merged.append(dashboard.dropdowns)
            dashboards_merged.append(dashboard.graph)

        if data_dict['TSNE']:
            main_stats.apply_tsne()
            dashboard = DimRedDash(stats=main_stats, method='TSNE', plot_options=data_dict['TSNE'])

            dashboards_merged.append(dashboard.title)
            dashboards_merged.append(dashboard.dropdowns)
            dashboards_merged.append(dashboard.graph)

        if data_dict['LLE']:
           main_stats.apply_lle()
           dashboard = DimRedDash(stats=main_stats, method='LLE', plot_options=data_dict['LLE'])

           dashboards_merged.append(dashboard.title)
           dashboards_merged.append(dashboard.dropdowns)
           dashboards_merged.append(dashboard.graph)


        if data_dict['UMAP']:
           main_stats.apply_umap()
           dashboard = DimRedDash(stats=main_stats, method='UMAP', plot_options=data_dict['UMAP'])
           dashboards_merged.append(dashboard.title)
           dashboards_merged.append(dashboard.dropdowns)
           dashboards_merged.append(dashboard.graph)


        if data_dict['KMAP']:
           main_stats.apply_kmap()
           dashboard = DimRedDash(stats=main_stats, method='KMAP', plot_options=data_dict['KMAP'])
           dashboards_merged.append(dashboard.title)
           dashboards_merged.append(dashboard.dropdowns)
           dashboards_merged.append(dashboard.graph)

        if data_dict['ISOMAP']:
           main_stats.apply_isomap()
           dashboard = DimRedDash(stats=main_stats, method='ISOMAP', plot_options=data_dict['ISOMAP'])
           dashboards_merged.append(dashboard.title)
           dashboards_merged.append(dashboard.dropdowns)
           dashboards_merged.append(dashboard.graph)

        if data_dict['MDS']:
            main_stats.apply_mds()
            dashboard = DimRedDash(stats=main_stats, method='MDS', plot_options=data_dict['MDS'])
            dashboards_merged.append(dashboard.title)
            dashboards_merged.append(dashboard.dropdowns)
            dashboards_merged.append(dashboard.graph)

        # Merge
        # Merge all dashboards here
        selected_options_number = (len(dashboards_merged)-1)/2

        self.dash_app.css.config.serve_locally = False
        # Boostrap CSS.
        self.dash_app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})  # noqa: E501
        self.dash_app.index_string = html_layout

        self.dash_app.layout = html.Div(
            html.Div( children= dashboards_merged, className='twelve columns offset-by-one')
        )

        if data_dict['PCA']:
             if 'scatter' in data_dict['PCA']:
                 @self.dash_app.callback(
                     [Output(component_id='reduced_data_plot_pca', component_property='figure')],
                     [Input(component_id='red_dim_input_pca', component_property='value'),
                      Input(component_id='outlier_only_options_pca', component_property='value')]
                 )

                 def update_pca(m, show_only_outl_option):
                     print(data_dict)
                     main_stats.apply_pca(m=m)
                     if show_only_outl_option:
                         pca_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_pca[
                             main_stats.reduced_pandas_dataframe_pca['Classification'] == 'Outliers']).plot_data()
                     else:
                         pca_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_pca).plot_data()

                     return [pca_vis]

        if data_dict['LLE']:

             if 'scatter' in data_dict['LLE']:
                 @self.dash_app.callback(
                     [Output(component_id='reduced_data_plot_lle', component_property='figure')],
                     [Input(component_id='red_dim_input_lle', component_property='value'),
                      Input(component_id='neighbours_lle', component_property='value'),
                      Input(component_id='outlier_only_options_lle', component_property='value')
                 ]
                 )

                 def update_lle(m, k_neighbours, show_only_outl_option):
                     main_stats.apply_lle(m=m, k=k_neighbours)

                     if show_only_outl_option:
                         lle_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_lle[
                             main_stats.reduced_pandas_dataframe_lle['Classification'] == 'Outliers']).plot_data()
                     else:
                         lle_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_lle).plot_data()

                     return [lle_vis]

        if data_dict['TSNE']:

             if 'scatter' in data_dict['TSNE']:
                 @self.dash_app.callback(
                     [Output(component_id='reduced_data_plot_tsne', component_property='figure')],
                     [Input(component_id='red_dim_input_tsne', component_property='value'),
                      Input(component_id='perplexity_tsne', component_property='value'),
                      Input(component_id='outlier_only_options_tsne', component_property='value')
                      ]
                 )

                 def update_tsne(m, perplexity, show_only_outl_option):
                     main_stats.apply_tsne(m=m, perplexity=perplexity)
                     if show_only_outl_option:
                         tsne_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_tsne[
                             main_stats.reduced_pandas_dataframe_tsne['Classification'] == 'Outliers']).plot_data()
                     else:
                         tsne_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_tsne).plot_data()

                     return [tsne_vis]


        if data_dict['UMAP']:

             if 'scatter' in data_dict['UMAP']:
                 @self.dash_app.callback(
                     [Output(component_id='reduced_data_plot_umap', component_property='figure')],
                     [Input(component_id='red_dim_input_umap', component_property='value'),
                      Input(component_id='kneighbours_umap', component_property='value'),
                      Input(component_id='outlier_only_options_umap', component_property='value')]
                 )

                 def update_umap(m, k_neighbours, show_only_outl_option):
                     main_stats.apply_umap(m=m, k=k_neighbours)
                     if show_only_outl_option:
                         umap_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_umap[
                             main_stats.reduced_pandas_dataframe_umap['Classification'] == 'Outliers']).plot_data()
                     else:
                         umap_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_umap).plot_data()

                     return [umap_vis]


        if data_dict['KMAP']:

             if 'scatter' in data_dict['KMAP']:
                 @self.dash_app.callback(
                     [Output(component_id='reduced_data_plot_kmap', component_property='figure')],
                     [Input(component_id='red_dim_input_kmap', component_property='value'),
                      Input(component_id='kneighbours_kmap', component_property='value'),
                      Input(component_id='algs_kmap', component_property='value'),
                      Input(component_id='outlier_only_options_kmap', component_property='value')]
                 )

                 def update_kmap(m, k_neighbours, a, show_only_outl_option):
                     main_stats.apply_kmap(m=m, k=k_neighbours, a=a)

                     if show_only_outl_option:
                         kmap_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_kmap[
                             main_stats.reduced_pandas_dataframe_kmap['Classification'] == 'Outliers']).plot_data()
                     else:
                         kmap_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_kmap).plot_data()

                     return [kmap_vis]


        if data_dict['ISOMAP']:

             if 'scatter' in data_dict['ISOMAP']:
                 @self.dash_app.callback(
                     [Output(component_id='reduced_data_plot_isomap', component_property='figure')],
                     [Input(component_id='red_dim_input_isomap', component_property='value'),
                      Input(component_id='kneighbours_isomap', component_property='value'),
                      Input(component_id='outlier_only_options_isomap', component_property='value')
                      ]
                 )

                 def update_isomap(m, k_neighbours, show_only_outl_option):
                     main_stats.apply_isomap(m=m, k=k_neighbours)

                     if show_only_outl_option:
                         isomap_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_isomap[
                             main_stats.reduced_pandas_dataframe_isomap['Classification'] == 'Outliers']).plot_data()
                     else:
                         isomap_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_isomap).plot_data()

                     return [isomap_vis]


        if data_dict['MDS']:

             if 'scatter' in data_dict['MDS']:
                 @self.dash_app.callback(
                     [Output(component_id='reduced_data_plot_mds', component_property='figure')],
                     [Input(component_id='red_dim_input_mds', component_property='value'),
                      Input(component_id='outlier_only_options_mds', component_property='value')
                      ]
                 )

                 def update_isomap(m, show_only_outl_option):
                     main_stats.apply_mds(m=m)

                     if show_only_outl_option:
                         mds_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_mds[
                             main_stats.reduced_pandas_dataframe_mds['Classification'] == 'Outliers']).plot_data()
                     else:
                         mds_vis = VisualizationPlotly(pd_data_frame=main_stats.reduced_pandas_dataframe_mds).plot_data()

                     return [mds_vis]
        return self.dash_app.server



#dashboard_config = {'location': session['filename'],
#                    'target': alg_form.target.data,
#                    'PCA': [],
#'LLE':['scatter,'box','kn']
 #                   }




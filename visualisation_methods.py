import numpy as np
import plotly.express as px
import plotly.graph_objects as go

class VisualizationPlotly():
    ''' This class gives the user the possibility to make different plotly plots to visualize
        an input pandas data frame.
    '''

    def __init__(self, pd_data_frame):
        '''
        Input: pd_data_frame  - pandas data frame representing data
        '''
        self.pd_data_frame = pd_data_frame
        # Extract information about the data frame
        self.features = self.pd_data_frame.keys().tolist()

        if 'Classification' in self.features:
            # Read dimensions of the data
            self.d = self.pd_data_frame.shape[1] - 1
            self.features.remove('Classification')
            self.classification = True
        else:
            self.d = self.pd_data_frame.shape[1]
            self.classification = False
        # Read number of samples/examples of the data frame
        self.n = self.pd_data_frame.shape[0]

    def plot_data(self):
        ''' Visualize reduced data in a 1dim,  2dim or 3dim scatter plot. If the panda data frame contains "Classification"
        as one column, the plots are labeled, otherwise not.
        '''

        # If reduced data is just 1 dimensional
        if self.d == 1:
            if self.classification:
                fig = px.scatter(self.pd_data_frame, x=self.features[0], y=np.zeros(self.n),
                                 color='Classification', title='Data')
            else:
                fig = px.scatter(self.pd_data_frame, x=self.features[0], y=np.zeros(self.n), c='blue', title='Data')

        elif self.d == 2:
            if self.classification:
                fig = px.scatter(self.pd_data_frame, x=self.features[0], y=self.features[1],
                                 color='Classification', title='Data')
            else:
                fig = px.scatter(self.pd_data_frame, x=self.features[0], y=self.features[1], c='blue', title='Data')

        else:
            if self.classification:
                fig = px.scatter_3d(self.pd_data_frame, x=self.features[0], y=self.features[1], z=self.features[2],
                                    color='Classification', title='Data')
            else:
                fig = px.scatter_3d(self.pd_data_frame, x=self.features[0], y=self.features[1], z=self.features[2],
                                    color='blue', title='Data')

        return fig

    def box_plot_classifications(self, dim):
        ''' This method is only to be used if the pandas dataframe is labeled. It displays the statistical information
        in a boxplot for an input feature (e.g if 2dim data how the inliers/outliers are distributed along the 1 dimension)

        Input: dim    - int, dimension for which the statistical info is shown
        '''
        if self.classification:
            fig = px.box(self.pd_data_frame, x="Classification", y=self.features[dim], points="all",
                         title='Statistical Information')
        else:
            print('Error: Data is not Labeled')
        return fig

    def histogram_data(self, dim):
        ''' Plot an histogram for an given dim/feature of the pandas data frame. If the data is labeled, there is one
            histogram for each class, otherwise not.

            Input: dim    - int, dimension for which the histogram is computed
        '''

        if self.classification:
            fig = px.histogram(self.pd_data_frame, x=self.features[dim], color="Classification", title='Histogram',
                               nbins=30, marginal="rug", opacity=0.7)
        else:
            fig = px.histogram(self.pd_data_frame, x=self.features[dim], title='Histogram', nbins=30, marginal="rug",
                               opacity=0.7)

        return fig

    def graph_neighbours(self, edges, nodes):

        # Check if the graph is 2dim or 3dim
        graph_dim = len(nodes[0])

        if graph_dim == 2:
            edge_x = []
            edge_y = []
            for edge in edges:
                x0, y0 = nodes[edge[0]]
                x1, y1 = nodes[edge[1]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []
            for node in nodes:
                x, y = node
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    # colorscale options
                    # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                    # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                    # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                    colorscale='YlGnBu',
                    reversescale=True,
                    color=[],
                    size=10,
                    line_width=2))


        elif graph_dim == 3:
            edge_x = []
            edge_y = []
            edge_z = []
            for edge in edges:
                x0, y0, z0 = nodes[edge[0]]
                x1, y1, z1 = nodes[edge[1]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
                edge_z.append(z0)
                edge_z.append(z1)
                edge_z.append(None)


            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []
            node_z = []
            for node in nodes:
                x, y, z = node
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)

            node_trace = go.Scatter3d(
                x=node_x, y=node_y, z = node_z,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    # colorscale options
                    # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                    # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                    # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                    colorscale='YlGnBu',
                    reversescale=True,
                    color=[],
                    size=10,
                    line_width=4))

        fig = go.Figure(data=[edge_trace, node_trace],
                              layout=go.Layout(
                                  title='<br>Network graph',
                                  titlefont_size=16,
                                  showlegend=False,
                                  hovermode='closest',
                                  margin=dict(b=20, l=5, r=5, t=40),
                                  annotations=[dict(
                                      text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                                      showarrow=False,
                                      xref="paper", yref="paper",
                                      x=0.005, y=-0.002)],
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                              )

        return fig

    def pie_plot_percentages(self):
        ''' Returns a pie plot chart figure showing how many datapoints % are outliers and how many datapoints are i
        inliers
        '''
        fig = px.pie(self.pd_data_frame, values=self.features[0],
                                     names='Classification', title='Outlier-Percentage Information')
        return fig



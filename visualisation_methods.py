import numpy as np
import plotly.express as px




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
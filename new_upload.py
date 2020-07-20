import numpy as np
import pandas as pd
# The main idea is that the user is able to upload a csv dataframe that is not labeled with Outliers/Inliers.
# The csv dataframe should contain some kind of labels e.g in the mnist case the labels 0-9 or in the mnist
# case the label 'shoes' 'thshirt' 'pullover' etc.
import numpy as np
import pandas as pd
from statistics_methods import DataStatistics
from visualisation_methods import VisualizationPlotly
import matplotlib.pyplot as plt

# The main idea is that the user is able to upload a csv dataframe that is not labeled with Outliers/Inliers.
# The csv dataframe should contain some kind of labels e.g in the mnist case the labels 0-9 or in the mnist
# case the label 'shoes' 'thshirt' 'pullover' etc.

# Upload csv dataframe
file_name = '/Users/albertorodriguez/Desktop/Current Courses/InternenServLab/EVA/data/fmnist.csv'

# The Last Column of the data frame needs to contain the labels

main = DataStatistics()
main.load_data(file_name=file_name)

main.label_column = 'Clothes'

print(main.pandas_data_frame['Clothes'].unique())

clothes = list(main.pandas_data_frame['Clothes'].unique())

'''
for index, cur_cloth in enumerate(clothes):
    print(cur_cloth)
    print('the next clothe appears: {}, times'.format(main.pandas_data_frame[main.pandas_data_frame['Clothes'] == cur_cloth].shape[0]))
    cloth_pd = main.pandas_data_frame_nolabels[main.pandas_data_frame['Clothes'] == cur_cloth]
    cloth_pd = cloth_pd.to_numpy()
    image = cloth_pd[0, 1:-1].reshape((28, 28))
    image = image.astype('int')
    print(type(image))
    plt.figure(num=cur_cloth)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

plt.show()
'''

main.inliers = ['Sneaker']

main.outliers = ['Pullover']
main.ratio = 1
main.create_labeled_df()

main.apply_umap(m=2, k=15)
#main.apply_pca(m=3)
#main.apply_tsne(m=3)
#print(main.remained_variance)
fig = VisualizationPlotly(pd_data_frame=main.reduced_pandas_dataframe_umap).plot_data()

fig.show()
#main.apply_pca(m=4)

#print(main.pandas_data_frame[main.pandas_data_frame['Classification'] == 'Outliers'].shape)
#replace = main.reduced_pandas_dataframe_pca
#replace = replace.drop(['Classification'], axis=1)
#main.pandas_data_frame_nolabels = replace
#print(replace)
#print('Remained Variance: {}'.format(main.remained_variance))

#main.apply_tsne()

#fig = VisualizationPlotly(pd_data_frame=main.reduced_pandas_dataframe_tsne).plot_data()

#fig.show()

#['Ankle boot' 'T-shirt/top' 'Dress' 'Pullover' 'Sneaker' 'Sandal'
# 'Trouser' 'Shirt' 'Coat' 'Bag']
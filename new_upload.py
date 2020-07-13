import numpy as np
import pandas as pd
# The main idea is that the user is able to upload a csv dataframe that is not labeled with Outliers/Inliers.
# The csv dataframe should contain some kind of labels e.g in the mnist case the labels 0-9 or in the mnist
# case the label 'shoes' 'thshirt' 'pullover' etc.

# Upload csv dataframe
file_name = './data/fmnist.csv'
pandas_df = pd.read_csv(file_name)

# The Last Column of the data frame needs to contain the labels
labels_column_name = pandas_df.columns[-1]
labels = pandas_df.iloc[:, -1]
np_labels = np.asarray(labels)

# Read dimensions of the dataset
n, d = pandas_df.shape
d -= 1  # cause labels take one extra dim in the formula above

# See how much each label occurs in the data frame
labels_unique = np.unique(np_labels)

# the user can read from labels unique which he wishes to be the outliers and which the inliers

labels_percentage = np.zeros_like(labels_unique)

for index, cur_label in enumerate(labels_unique):
    labels_percentage[index] = len(np.argwhere(np_labels == cur_label))/n * 100

# We can display labels_percentage in the homepage or some other window

# E.G User selects in fmnist 5 as inliers and 8 as Outliers
# Moreover, the user can enter which % of outliers in the dataset he wishes to have
selected_inlier = 5
selected_outlier = 8
outlier_selected_percentage = 1
outlier_selected_percentage = outlier_selected_percentage/100

Outliers_pd = pandas_df[pandas_df[str(labels_column_name)] == selected_outlier]
Inliers_pd = pandas_df[pandas_df[str(labels_column_name)] == selected_inlier]

N_inl = Inliers_pd.shape[0]
N_outl = int((N_inl*outlier_selected_percentage)/(1 - outlier_selected_percentage))

Outliers_pd_final = Outliers_pd[0:N_outl]

Outliers_pd_final[str(labels_column_name)] = 'Outliers'
Inliers_pd[str(labels_column_name)] = 'Inlier'

final_df = pd.concat([Inliers_pd, Outliers_pd_final], ignore_index=True)


# Store data frame for later use in the app
store_name = 'fmnist_example'
final_df.to_csv(store_name, index=False)
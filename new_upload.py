import numpy as np
import pandas as pd
# The main idea is that the user is able to upload a csv dataframe that is not labeled with Outliers/Inliers.
# The csv dataframe should contain some kind of labels e.g in the mnist case the labels 0-9 or in the mnist
# case the label 'shoes' 'thshirt' 'pullover' etc.
import numpy as np
import pandas as pd

# The main idea is that the user is able to upload a csv dataframe that is not labeled with Outliers/Inliers.
# The csv dataframe should contain some kind of labels e.g in the mnist case the labels 0-9 or in the mnist
# case the label 'shoes' 'thshirt' 'pullover' etc.

# Upload csv dataframe
file_name = '/Users/albertorodriguez/Desktop/Current Courses/InternenServLab/EVA/data/mnist_train.csv'
pandas_df = pd.read_csv(file_name)


# The Last Column of the data frame needs to contain the labels


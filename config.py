external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

iris_config = {'location': 'application/data/iris_anomalies.csv',
               'features': ['is_anomaly', 'petal.length', 'petal.width','sepal.length'],
               'target': 'variety'}


app_secret_key = 'b042acdb071518100e25f40b93088487'

session = {}

vis_types = ['Scatter', 'Box', 'K-Nearest']
alg_types = ['PCA', 'LLE', 'TSNE']

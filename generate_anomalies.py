import argparse
import pandas as pd
import numpy as np
# TODO currently hardcoded to iris, extend to other sources, extend options

parser = argparse.ArgumentParser()
parser.add_argument('--output', help='Name of the output csv file.',
                    type=str, required=True)
args = parser.parse_args()
df = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv')
targets = df['Name'].unique().tolist()
not_anomaly = np.random.choice(targets)
anomalies = targets.remove(not_anomaly)
new_rows = []
for row in df.to_dict('records'):
    if row['Name'] == not_anomaly:
        row.update({'is_anomaly': False})
        new_rows.append(row)
    elif np.random.uniform() <= 0.1:
        row.update({'is_anomaly': True})
        new_rows.append(row)

pd.DataFrame(new_rows).to_csv(args.output)

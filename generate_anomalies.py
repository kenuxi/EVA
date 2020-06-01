import argparse
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', help='Name of the output csv file.',
                    type=str, required=True)
parser.add_argument('-s', '--source', help='Link to the csv file.',
                    type=str, required=True)
parser.add_argument('-t', '--target', help='Name of the target column.',
                    type=str, required=True)
parser.add_argument('-p', help='Probability of creating anomalies.',
                    type=float, required=False)
args = parser.parse_args()

try:
    df = pd.read_csv(args.source)
except FileNotFoundError:
    raise FileNotFoundError(f'Source ({args.source}) not found!')

# TODO handle cases when columns are not named
try:
    targets = df[args.target].unique().tolist()
except KeyError:
    raise KeyError(f'Target column ({args.target}) not found!')

p = 0.05 if not args.p else args.p
not_anomaly = np.random.choice(targets)
anomalies = np.copy(targets).tolist()
anomalies.remove(not_anomaly)
new_rows = []

df_dict = df.to_dict('records')  # This line make take a while

for idx, row in enumerate(df_dict):
    if row[args.target] == not_anomaly:
        row.update({'is_anomaly': False})
        new_rows.append(row)
    elif np.random.uniform() <= p:
        row.update({'is_anomaly': True})
        new_rows.append(row)
pd.DataFrame(new_rows).to_csv(args.output)

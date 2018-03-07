from data.raw_data import *
import pandas as pd

average2 = base_dir + 'average2\\'
sub_files = [
'average(1).csv',
 'average(2).csv',
 'average(3).csv',
 'average(4).csv',
 'average(5).csv',
 'average(6).csv',
 'average.csv',
 'average2.csv',
 'average5.csv'
]

sub_dfs = [pd.read_csv(average2 + sub_file) for sub_file in sub_files]

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

for sub_df in sub_dfs:
    for label in labels:
        max_value = sub_df[label].max()
        min_value = sub_df[label].min()
        sub_df[label] = (sub_df[label] - min_value) / (max_value - min_value)

sub = sub_dfs[0]

for i in range(1, len(sub_dfs)):
    for label in labels:
        sub[label] = sub[label] + sub_dfs[i][label]

for label in labels:
    sub[label] = sub[label] / len(sub_dfs)

sub.to_csv(base_dir + 'average.csv', index=False)
from data.raw_data import *
import pandas as pd

base_dir = base_dir + 'average\\'
sub_files = [
'bidirectional_lstm.csv',
 'cnn_gru.csv',
 'cnn_submission.csv',
 'lr_words_and_char_ngrams.csv',
 'pooled_gru.csv',
 'pooled_gru_9823.csv',
 'pooled_gru_trainable.csv',
 'pooled_gru_trainable2(1).csv',
 'pooled_gru_trainable2.csv',
 'pooled_lstm_trainable.csv'
]

sub_dfs = [pd.read_csv(base_dir + sub_file) for sub_file in sub_files]

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

sub.to_csv(base_dir + 'average5.csv', index=False)
import pandas as pd
import os

base_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Toxic Comment Classification Challenge\\'

train_file_name = 'train.csv'
test_file_name = 'test.csv'
submission_file_name = 'sample_submission.csv'

train_df = pd.read_csv(base_dir + train_file_name)
test_df = pd.read_csv(base_dir + test_file_name)
submission = pd.read_csv(base_dir + submission_file_name)

statistics_columns = ['capitals', 'caps_vs_length']
if os.path.exists(base_dir + 'train_statistics.csv'):
    train_statistics = pd.read_csv(base_dir + 'train_statistics.csv')[statistics_columns].fillna(0).values
    test_statistics = pd.read_csv(base_dir + 'test_statistics.csv')[statistics_columns].fillna(0).values

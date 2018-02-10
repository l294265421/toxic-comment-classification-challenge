import pandas as pd

base_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Toxic Comment Classification Challenge\\'

train_file_name = 'train.csv'
test_file_name = 'test.csv'

train_df = pd.read_csv(base_dir + train_file_name)
test_df = pd.read_csv(base_dir + test_file_name)
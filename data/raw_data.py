import pandas as pd
from nltk import word_tokenize

base_dir = r'D:\document\program\ml\machine-learning-databases\kaggle\Toxic Comment Classification Challenge\\'

train_file_name = 'train.csv'
test_file_name = 'test.csv'
submission_file_name = 'sample_submission.csv'

train_df = pd.read_csv(base_dir + train_file_name)
test_df = pd.read_csv(base_dir + test_file_name)
submission = pd.read_csv(base_dir + submission_file_name)

train_df['word_num'] = train_df.apply(lambda row : len(word_tokenize(row['comment_text'])), axis=1)
test_df['word_num'] = test_df.apply(lambda row : len(word_tokenize(row['comment_text'])), axis=1)
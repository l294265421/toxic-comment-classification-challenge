from data.raw_data import *
from sklearn.linear_model import LogisticRegression
from data.raw_data import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk import pos_tag, word_tokenize
from scipy.sparse import hstack

all_comment_text = pd.concat([train_df[['comment_text']], test_df[['comment_text']]], axis=0)

all_comment_text['word_num'] = all_comment_text.apply(lambda row : len(word_tokenize(row['comment_text'])), axis=1)
all_comment_text['total_length'] = all_comment_text['comment_text'].apply(len)
all_comment_text['capitals'] = all_comment_text['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
all_comment_text['caps_vs_length'] = all_comment_text.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
all_comment_text['num_exclamation_marks'] = all_comment_text['comment_text'].apply(lambda comment: comment.count('!'))
all_comment_text['num_question_marks'] = all_comment_text['comment_text'].apply(lambda comment: comment.count('?'))
all_comment_text['num_punctuation'] = all_comment_text['comment_text'].apply(
    lambda comment: sum(comment.count(w) for w in '.,;:'))
all_comment_text['num_symbols'] = all_comment_text['comment_text'].apply(
    lambda comment: sum(comment.count(w) for w in '*&$%'))
all_comment_text['num_words'] = all_comment_text['comment_text'].apply(lambda comment: len(comment.split()))
all_comment_text['num_unique_words'] = all_comment_text['comment_text'].apply(
    lambda comment: len(set(w for w in comment.split())))
all_comment_text['words_vs_unique'] = all_comment_text['num_unique_words'] / all_comment_text['num_words']
all_comment_text['num_smilies'] = all_comment_text['comment_text'].apply(
    lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))

all_comment_text.fillna(0)
all_comment_text = all_comment_text.drop('comment_text', axis=1)

# 规范化数值型数据
for numerical_column in all_comment_text:
     column = all_comment_text[numerical_column]
     mean = column.mean()
     std = column.std()
     all_comment_text[numerical_column] = (all_comment_text[numerical_column] - mean) / std

train_statistics = all_comment_text[:len(train_df)]
train_statistics.to_csv(base_dir + 'train_statistics.csv', index=False)
test_statistics = all_comment_text[len(train_df):]
test_statistics.to_csv(base_dir + 'test_statistics.csv', index=False)

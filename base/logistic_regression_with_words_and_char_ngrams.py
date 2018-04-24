from sklearn.linear_model import LogisticRegression
from data.raw_data import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_text = train_df['comment_text']
test_text = test_df['comment_text']
all_text = pd.concat([train_text, test_text])

english_stemmer = SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

word_vectorizer = StemmedTfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 5),
    max_features=30000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

X = hstack([train_char_features, train_word_features], format='csr')
X_test = hstack([test_char_features, test_word_features], format='csr')

from sklearn.model_selection import KFold

result = []
ntrain = X.shape[0]
oof_train = np.zeros((ntrain, 6))
k = 4
kf = KFold(n_splits=k, shuffle=False)
for train_index, test_index in kf.split(X):
    aucs = []
    test_temp = test_df.copy()
    oof_train_index = 0
    for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        y = train_df[label]
        X_train = X[train_index]
        y_train = y[train_index]
        X_validation = X[test_index]
        y_validation = y[test_index]
        model = LogisticRegression(random_state=1234)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_validation)[:, 1]
        oof_train[test_index, oof_train_index] = y_pred
        oof_train_index += 1
        aucs.append(roc_auc_score(y_validation, y_pred))
        test_temp[label] = model.predict_proba(X_test)[:, 1]
    test_temp = test_temp.drop('id', axis=1)
    test_temp = test_temp.drop('comment_text', axis=1)
    result.append(test_temp.values)
    print(aucs)
    print('mean:{m}'.format(m=(sum(aucs) / len(aucs))))

y_test = result[0]
for i in range(1, k):
    y_test += result[i]
y_test /= k

train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = oof_train
train_df[['id', "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].to_csv(base_dir + 'logistic_train.csv', index=False)

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test
submission.to_csv(base_dir + 'logistic.csv', index=False)
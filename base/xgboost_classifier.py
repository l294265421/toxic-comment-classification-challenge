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
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

list_stopWords=list(set(stopwords.words('english')))
list_stopWords.append('')

english_stemmer = SnowballStemmer('english')

def normalize_word(word):
    word = word.strip('\'')
    if word.isdigit():
        return 'num'
    elif len(word) > 15:
        return 'execeptionword'.strip()
    elif len(word) < 3:
        return ''
    else:
        return english_stemmer.stem(word)


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (normalize_word(w) for w in analyzer(doc))


def lemmatize_and_stem_all(sentence):
    wnl = WordNetLemmatizer()
    for word in word_tokenize(sentence):
        yield wnl.lemmatize(word)


class LemmatizeTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        return lambda doc: lemmatize_and_stem_all(doc)


train_text = train_df['comment_text'].str.replace(r"[!\"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]", " ")
test_text = test_df['comment_text'].str.replace(r"[!\"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]", " ")
all_text = pd.concat([train_text, test_text])

v = StemmedTfidfVectorizer(stop_words=list_stopWords, tokenizer=word_tokenize, ngram_range=(1, 1), max_features=15000, max_df=0.5, min_df=0.00001)
v.fit(all_text)
X1 = v.transform(train_df['comment_text'])
X1_test = v.transform(test_df['comment_text'])

v = StemmedTfidfVectorizer(tokenizer=word_tokenize, ngram_range=(2, 3), max_features=5000, max_df=0.5, min_df=0.00001)
v.fit(all_text)
X2 = v.transform(train_df['comment_text'])
X2_test = v.transform(test_df['comment_text'])

char_vectorizer = TfidfVectorizer(
    lowercase=False,
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(2, 5),
    max_features=10000, max_df=0.5, min_df=0.00001)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

X = hstack([X1, X2, train_char_features])
X_test = hstack([X1_test, X2_test, test_char_features])

aucs = []
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    y = train_df[label]
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25, random_state=1234)
    model = XGBClassifier(n_estimators=400,n_jobs=2, learning_rate=0.1, booster='gbtree', random_state=1234,
                          max_depth=6, subsample=0.8, colsample_bytree=0.8)
    model.fit(X_train, y_train)
    aucs.append(roc_auc_score(y_validation, model.predict_proba(X_validation)[:, 1]))
    test_df[label] = model.predict_proba(X_test)[:, 1]
print(aucs)
print('mean:{m}'.format(m=(sum(aucs)/len(aucs))))

test_df.drop('comment_text', axis=1, inplace=True)
test_df.to_csv(base_dir + 'xgboost.csv', index=False)
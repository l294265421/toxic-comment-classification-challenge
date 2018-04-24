from sklearn.ensemble import RandomForestClassifier
from data.raw_data import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer

english_stemmer = SnowballStemmer('english')
def normalize_word(word):
    if word.isdigit():
        return 'num'
    elif len(word) > 15:
        return 'execeptionword'
    else:
        return english_stemmer.stem(word)

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (normalize_word(w) for w in analyzer(doc))

v = StemmedTfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=15000)

X = v.fit_transform(train_df['comment_text'])
print(str(len(v.vocabulary_)))
X_test = v.transform(test_df['comment_text'])

from sklearn.model_selection import KFold
import numpy as np

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
        model = RandomForestClassifier(n_jobs=4, n_estimators=200)
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
train_df[['id', "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].to_csv(base_dir + 'rf_train.csv', index=False)

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test
submission.to_csv(base_dir + 'rf.csv', index=False)
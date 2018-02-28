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

aucs = []
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    y = train_df[label]
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.4, random_state=1234)
    model = RandomForestClassifier(n_jobs=4, n_estimators=100)
    model.fit(X_train, y_train)
    aucs.append(roc_auc_score(y_validation, model.predict_proba(X_validation)[:, 1]))
    test_df[label] = model.predict_proba(X_test)[:, 1]
print(aucs)
print('mean:{m}'.format(m=(sum(aucs)/len(aucs))))

test_df.drop('comment_text', axis=1, inplace=True)
test_df.to_csv(base_dir + 'rf.csv', index=False)
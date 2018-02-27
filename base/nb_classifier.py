from sklearn.linear_model import LogisticRegression
from data.raw_data import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import SnowballStemmer
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

english_stemmer = SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def lemmatize_and_stem_all(sentence):
    wnl = WordNetLemmatizer()
    for word in word_tokenize(sentence):
        yield wnl.lemmatize(word)

class LemmatizeTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        return lambda doc: lemmatize_and_stem_all(doc)

v = StemmedTfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_df=0.5, max_features=30000)

X = v.fit_transform(train_df['comment_text'])
X_test = v.transform(test_df['comment_text'])

aucs = []
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    y = train_df[label]
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.4, random_state=1234)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    aucs.append(roc_auc_score(y_validation, model.predict_proba(X_validation)[:, 1]))
    test_df[label] = model.predict_proba(X_test)[:, 1]
print(aucs)
print('mean:{m}'.format(m=(sum(aucs)/len(aucs))))

test_df.drop('comment_text', axis=1, inplace=True)
test_df.to_csv(base_dir + 'nb.csv', index=False)

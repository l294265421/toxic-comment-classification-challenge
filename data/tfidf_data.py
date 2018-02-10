from data.raw_data import *
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer()

X_train = v.fit_transform(train_df['comment_text'])
X_test = v.transform(test_df['comment_text'])

labels = {}

for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    labels[label] = train_df[label]
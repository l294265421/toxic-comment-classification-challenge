from data.tfidf_data import *
from sklearn.linear_model import LogisticRegression
from  data.raw_data import *

for label in labels.keys():
    y = labels[label]
    model = LogisticRegression()
    model.fit(X_train, y)
    test_df[label] = model.predict_proba(X_test)[:, 1]

test_df.drop('comment_text', axis=1, inplace=True)
test_df.to_csv(base_dir + 'logistic.csv', index=False)

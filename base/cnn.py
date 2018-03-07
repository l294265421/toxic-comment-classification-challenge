import numpy as np

np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping

import warnings

warnings.filterwarnings('ignore')

import os

import gensim

from data.raw_data import *

os.environ['OMP_NUM_THREADS'] = '4'

EMBEDDING_FILE = base_dir + 'crawl-300d-2M.vec'

train = train_df
test = test_df
submission = submission

X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values

max_features = 30000
maxlen = 300
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

word_index = tokenizer.word_index

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding='utf-8') if o.strip().split()[0] in word_index)

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

nb_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
def get_model():
    inp = Input(shape=(maxlen,))
    embedded_sequences = Embedding(nb_words,
                                embed_size,
                                weights=[embedding_matrix],
                                trainable=True)(inp)

    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs)
    x = Dropout(0.5)(l_merge)
    # Finally, we feed the output into a Sigmoid layer.
    # The reason why sigmoid is used is because we are trying to achieve a binary classification(1,0)
    # for each of the 6 labels, and the sigmoid function will squash the output between the bounds of 0 and 1.
    preds = Dense(6, activation='sigmoid')(x)

    model = Model(inp, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model

model = get_model()

from sklearn.model_selection import KFold

result = []
k = 4
kf = KFold(n_splits=k, shuffle=False)
for train_index, test_index in kf.split(x_train):
    X_tra = x_train[train_index]
    y_tra = y_train[train_index]
    X_val = x_train[test_index]
    y_val = y_train[test_index]

    batch_size = 32
    epochs = 1
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
    callbacks_list = [early_stopping, RocAuc]

    hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                     callbacks=callbacks_list, verbose=2)

    y_pred = model.predict(x_test, batch_size=1024)
    result.append(y_pred)

y_test = result[0]
for i in range(1, k):
    y_test += result[i]
y_test /= k

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test
submission.to_csv(base_dir + 'cnn.csv', index=False)
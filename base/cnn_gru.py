'''https://www.kaggle.com/kashyap32/keras-cnn-gru/notebook'''

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input,GRU, LSTM
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D, Dropout, concatenate
from keras.preprocessing import text, sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint

from data.raw_data import *

max_features = 20000
maxlen = 100

train = train_df
test = test_df
print(train.head(10))
list_sentences_train = train["comment_text"].fillna("unknown").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("unknown").values

print(list_sentences_train[0])
y[0]

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
# train data
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
# test data
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

def cnn_rnn():
    embed_size = 256
    inp = Input(shape=(maxlen, ))
    main = Embedding(max_features, embed_size)(inp)
    main = Dropout(0.2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = GRU(32)(main)
    main = Dense(16, activation="relu")(main)
    main = Dense(6, activation="sigmoid")(main)
    model = Model(inputs=inp, outputs=main)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

model = cnn_rnn()
model.summary()

from sklearn.model_selection import train_test_split

print('Positive Labels ')
any_category_positive = np.sum(y,1)
print(pd.value_counts(any_category_positive))
X_t_train, X_t_test, y_train, y_test = train_test_split(X_t, y,
                                                        test_size = 0.10,
                                                        )
print('Training:', X_t_train.shape)
print('Testing:', X_t_test.shape)

batch_size = 128
epochs = 3

file_path="model_best.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early] #early
model.fit(X_t_train, y_train,
          validation_data=(X_t_test, y_test),
          batch_size=batch_size,
          epochs=epochs,
          shuffle = True,
          callbacks=callbacks_list)

model.save('Whole_model.h5')
model.load_weights(file_path)
y_test = model.predict(X_te)
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("predictions.csv", index=False)

#For fun not for testing purpose!!
test=["This is your last warning. You will be blocked from editing the next time you vandalize a page, as you did with this edit to Geb.  |Parlez ici "]

tokenizer.fit_on_texts(list(test))
# train data
test_token = tokenizer.texts_to_sequences(test)
test_2 = sequence.pad_sequences(test_token, maxlen=maxlen)

np.argmax(model.predict(test_2))

model.predict(test_2)

pred=pd.read_csv('predictions.csv')
pred.head()






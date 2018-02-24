from data.raw_data import *
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os

stop_words = set(stopwords.words('english'))
english_stemmer = SnowballStemmer('english')

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"\'", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def lemmatize_and_stem_all(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith("NN"):
            yield english_stemmer.stem(wnl.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            yield english_stemmer.stem(wnl.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            yield english_stemmer.stem(wnl.lemmatize(word, pos='a'))
        elif tag.startswith('R'):
            yield english_stemmer.stem(wnl.lemmatize(word, pos='r'))
        else:
            yield english_stemmer.stem(word)

def msgProcessing(raw_msg):
    m_w = []
    words2 = []
    raw_msg = str(raw_msg)
    raw_msg = str(raw_msg.lower())
    # url_stripper= re.sub(r'Email me.*[A-Z]',"",s)

    # raw_msg=re.sub(r'\w*[0-9]\w*','', url_stripper)
    raw_msg = re.sub(r'[^a-zA-Z]', ' ', raw_msg)

    words = raw_msg.lower().split()
    # Remove words with length lesser than 3 if not w in stops
    for i in words:
        if len(i) >= 2:
            words2.append(i)
    stops = set(stopwords.words('english'))
    m_w = " ".join([w for w in words2 if w not in stops])
    return (" ".join(lemmatize_and_stem_all(m_w)))


def helperFunction(df):
    print("Data Preprocessing!!!")
    cols = ['comment_text']
    df = df[cols]
    df.comment_text.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
    num_msg = df[cols].size
    clean_msg = []
    for i in range(0, num_msg):
        clean_msg.append(msgProcessing(df['comment_text'][i]))
    df['Processed_msg'] = clean_msg
    print("Data Preprocessing Ends!!!")
    return df

train_file_name = 'clean_train.csv'
test_file_name = 'clean_test.csv'

if os.path.exists(base_dir + train_file_name):
    train_df = pd.read_csv(base_dir + train_file_name)
    test_df = pd.read_csv(base_dir + test_file_name)
else:
    train_df = helperFunction(train_df)
    train_df.to_csv(base_dir + train_file_name, index=False)
    test_df = helperFunction(test_df)
    test_df.to_csv(base_dir + test_file_name, index=False)




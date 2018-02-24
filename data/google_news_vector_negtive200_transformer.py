from data.vocabulary import vocab
from  data.raw_data import *
import gensim
from nltk.stem import SnowballStemmer

word2vec_path = base_dir + "GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

with open(base_dir + 'GoogleNews-vectors-negative300.txt', mode='w') as output_file:
    pass
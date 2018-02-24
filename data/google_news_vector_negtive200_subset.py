from data.vocabulary import vocab
from  data.raw_data import *
import gensim
from nltk.stem import SnowballStemmer

word2vec_path = base_dir + "GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

with open(base_dir + 'GoogleNews-vectors-negative300-subset.txt', mode='w') as output_file:
    for word in vocab:
        if word in word2vec:
            vec = word2vec[word]
            word_and_vec = []
            word_and_vec.append(word)
            for num in vec:
                word_and_vec.append(num)
            word_and_vec_str = ' '.join(word_and_vec)
            output_file.write(word_and_vec_str + '\n')
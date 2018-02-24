from data.clean_data import *
from nltk import word_tokenize
import os

vocab_file_path = base_dir + 'vocab.txt'

vocab = set()

if os.path.exists(vocab_file_path):
    with open(vocab_file_path, encoding='utf-8') as vocab_file:
        lines = vocab_file.readlines()
        vocab.union(lines)
else:
    all_training_words = [word for comment in train_df['comment_text'] for word in word_tokenize(comment)]
    all_test_words = [word for comment in test_df["comment_text"] for word in word_tokenize(comment)]

    vocab = set(all_training_words + all_test_words)
    with open(vocab_file_path, 'w', encoding='utf-8') as vocab_file:
        for word in vocab:
            vocab_file.write(word + '\n')



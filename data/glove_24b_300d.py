from data.raw_data import base_dir

file_name = 'glove.42B.300d.txt'

with open(base_dir + file_name) as glove_file:
    print(glove_file.readline())
    print(len(glove_file.readline().split()))
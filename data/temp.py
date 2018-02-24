from data.raw_data import base_dir

with open(base_dir + 'GoogleNews-vectors-negative300.bin', encoding='utf-8') as f:
    print(f.readline())

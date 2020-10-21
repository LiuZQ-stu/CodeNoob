import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import jieba
from ast import literal_eval


char_emb_path = '/data2/zhaoqi/project1/CNNNERmodel/data/char_50d.vec'
data_file_path = '/data2/zhaoqi/project1/data/attr/ner_data_split_backup'
target_path = '/data2/zhaoqi/project1/CNNNERmodel/data/word_emb_mean.vec'

char_dict = {}
with open(char_emb_path, 'r') as emb:
	embeddings = emb.readlines()
	for embedding in tqdm(embeddings):
		flag = 0
		if embedding[0] == ' ':
			flag = 1
		embedding = embedding.strip().split()
		if flag:
			char_dict[' '] = np.array(list(map(np.float32, embedding)))
		else:
			char_dict[embedding[0]] = np.array(list(map(np.float32, embedding[1:])))


files = os.listdir(data_file_path)
for idx, name in enumerate(files):
	if 'id' in name:
		files.pop(idx)

word_embs = {}
for file in files:
	file_path = os.path.join(data_file_path, file)
	data = pd.read_csv(file_path, sep='\t')
	for rdx in tqdm(range(len(data))):
		seq = data.loc[rdx, 'x']
		seq = jieba.lcut(''.join(literal_eval(seq)))
		for word in seq:
			if not word_embs.get(word) or not word_embs.get(word).any():
				continue
			char_embs = np.array([char_dict[char_] for char_ in list(word) if char_ in char_dict.keys()])
			if char_embs == np.array([]):
				word_embs[word] = np.random.randn(50)
			else:
				word_embs[word] = np.mean(char_embs, axis = 0)

with open(target_path, 'w') as f:
	for (word, embed) in word_embs.items():
		f.write(word)
		f.write(' '.join(list(map(str, embed))))
		f.write('\n')
print('done!')
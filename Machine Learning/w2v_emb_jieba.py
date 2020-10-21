#-*- coding : utf-8-*-

from gensim.models import Word2Vec
import os
from tqdm import tqdm
import jieba

class w2vModel:
	def __init__(self, stc):

		self.w2v_model = None
		self.stc = stc

	def train(self, embed_size=50, window_size=4, workers=10, iter=8, **kwargs):

		kwargs["sentences"] = self.stc.MakeSentences()
		kwargs["min_count"] = 3
		kwargs["size"] = embed_size
		kwargs["hs"] = 1  # Hierarchical Softmax
		kwargs["sg"] = 1  # skip gram
		kwargs["workers"] = workers
		kwargs["window"] = window_size
		kwargs["iter"] = iter
		#alpha, min_alpha, iter
		
		print("Learning embedding vectors...")
		model = Word2Vec(**kwargs)
		print("Learning embedding vectors done!")
		if model.wv.__contains__('短袖'):
			print(model.most_similar('短袖', topn=5))
		if model.wv.__contains__('冬季'):
			print(model.most_similar('冬季', topn=5))

		self.w2v_model = model
		return model

	def get_embeddings(self):

		emb = {}
		alphabet = self.stc.getAlphabet()
		for item in tqdm(alphabet, desc='getting embedding'):
			if self.w2v_model.wv.__contains__(item):
				emb[item] = self.w2v_model.wv[item].astype('float16')

		return emb

class JiebaStc:
	def __init__(self, data_file_path, filesList):

		self.data_file_path = data_file_path
		self.alphabet = set()
		self.filesList = filesList

	def getFilePathList(self):

		'''for idx, name in enumerate(self.filesList):
			if 'id' in name:
				filesList.pop(idx)'''
		for idx, name in enumerate(self.filesList):
			filesList[idx] = os.path.join(self.data_file_path, name)
		return filesList

	def readFile(self, file_path):

		singleStc = []
		data = open(file_path, 'rb').readlines()
		for rdx in tqdm(range(1, len(data))):
			try:
				temp = data[rdx].decode('utf-8').strip()
			except:
				continue
			else:
				pass
			arg = temp.find('\t')
			seq = temp[:arg][1:-1].split(', ')
			for i in range(len(seq)):
				seq[i] = seq[i].strip('\'')
			seq = jieba.lcut(''.join(seq).replace(' ', ''))

			singleStc.append(seq)
			for char_ in seq:
				self.alphabet.add(char_)
		return singleStc

	def MakeSentences(self):

		sentences = []
		files = self.getFilePathList()
		jieba.enable_parallel(8)
		for file in files:
			print(file)
			sentences += self.readFile(file)
		jieba.disable_parallel()

		return sentences

	def getAlphabet(self):

		return self.alphabet

def saveEmb(vec_target_path, emb):

	file_path = os.path.join(vec_target_path, 'jieba_50d_query.vec')
	with open(file_path, 'w', encoding='utf8') as f:
		for (key, emb) in emb.items():
			f.write(key + ' ')
			f.write(' '.join(list(map(str, emb))))
			f.write('\n')

if __name__ == "__main__":

	data_file_path = '/data2/zhaoqi/project1/data/attr/ner_data_split_backup'
	vec_target_path = '/data2/zhaoqi/project1/CNNNERmodel/data'
	filesList = ['query_for_train.tsv', 'query_for_dev.tsv']
	stcCla = JiebaStc(data_file_path, filesList)
	model = w2vModel(stcCla)
	model.train()
	emb = model.get_embeddings()
	saveEmb(vec_target_path, emb)
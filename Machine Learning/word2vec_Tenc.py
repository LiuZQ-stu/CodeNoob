import numpy as np
from gensim.models import Word2Vec
import pandas as pd
import os

class w2vModel:
	def __init__(self):

		self.w2v_model = None
		self.ad = None
		self.sentences = self.get_sentences()
		self._embeddings = {}

	def get_sentences(self):
		train_file = 'E:/Liuzq/腾讯广告算法大赛/train_preliminary'
		test_file = 'E:/Liuzq/腾讯广告算法大赛/test'
		
		ad_train = os.path.join(train_file, 'ad.csv')
		click_log_train = os.path.join(train_file, 'click_log.csv')
		ad_test = os.path.join(test_file, 'ad.csv')
		click_log_test = os.path.join(test_file, 'click_log.csv')

		trn_ad = pd.read_csv(ad_train, header=0, encoding='utf8').drop(['ad_id','product_id','industry'], axis = 1)
		tst_ad = pd.read_csv(ad_test, header=0, encoding='utf8').drop(['ad_id','product_id','industry'], axis = 1)
		trn_click_log = pd.read_csv(click_log_train, header=0, encoding='utf8').drop(['time','click_times'], axis=1)
		tst_click_log = pd.read_csv(click_log_test, header=0, encoding='utf8').drop(['time','click_times'], axis=1)

		self.ad = trn_ad.append(tst_ad).drop_duplicates().sort_values(by='creative_id', ignore_index='True')
		click_log = trn_click_log.append(tst_click_log).sort_values(by='user_id', ignore_index='True')

		del trn_ad
		del tst_ad
		del trn_click_log
		del tst_click_log
		grouped = click_log.groupby('user_id')
		stcs = []
		for user in set(click_log['user_id']):
			stcs.append(list(map(str, grouped.get_group(user)['creative_id'])))

		return stcs

	def train(self, embed_size=150, window_size=128, workers=10, iter=8, **kwargs):

		kwargs["sentences"] = self.sentences
		kwargs["min_count"] = 1
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

		self.w2v_model = model
		return model

	def get_embeddings(self):

		self._embeddings = {}
		for ads in self.ad['creative_id']:
			self._embeddings[str(ads)] = self.w2v_model.wv[str(ads)]

		return self._embeddings

def getUserVec(advec, sentences):

	userVec = {}

	for idx, sentence in enumerate(sentences):
		if idx < 900000:
			userVec[str(idx + 1)] = np.mean(np.array([advec[item] for item in sentence]), axis=0)
		else:
			userVec[str(idx + 2100001)] = np.mean(np.array([advec[item] for item in sentence]), axis=0)
	
	return userVec


if __name__ == "__main__":

	model = w2vModel()
	model.train()
	embeddings = model.get_embeddings()
	np.save('fulladVec150.npy', embeddings)
	userVec = getUserVec(embeddings, model.sentences)
	np.save('fulluserVec150.npy', userVec)
import pandas as pd
import string
from collections import defaultdict
import os
import random
from tqdm import tqdm
import jieba
from multiprocessing import Process

class Walker:
	def __init__(self, n):

		self.length_of_query = n
		self.data_file = '../data/'
		self.group_by_id_path = os.path.join(self.data_file, 'group_by_ids.csv')
		self.group_by_query_path = os.path.join(self.data_file, 'group_by_query.csv')
		self.tab = self.__makeTable()
		self.relevance = defaultdict(float)

	def buildGraph(self, target):

		graph = {}
		path = os.path.join(self.data_file, str(self.length_of_query) + '_' + target + '.txt')
		lines = open(path, 'r').readlines()
		for line in lines:
			line = line.strip().split()
			graph[line[0]] = line[1:]

		return graph

	def simulateWalk(self, iter_all=5, iter_single=3, total_num=100000):

		g_id = self.buildGraph('id')
		g_query = self.buildGraph('query')
		num_top = 50
		myinf = -10000
		for it in range(iter_all):
			for query in tqdm(g_query.keys()):
				self._simulateWalk(g_id, g_query, query, 1, iter_single)

		iter_relevance = {key:score/iter_all for key, score in self.relevance.items() if score > 0}
		for querys in iter_relevance.keys():
			q1 = list(querys[0])
			q2 = list(querys[1])
			if sorted(q1) == sorted(q2):
				iter_relevance[querys] = myinf
			elif '男' in q1:
				tmp_q = q1
				tmp_q[tmp_q.index('男')] = '女'
				if sorted(tmp_q) == sorted(q2):
					iter_relevance[querys] = myinf
			elif '男' in q2:
				tmp_q = q2
				tmp_q[tmp_q.index('男')] = '女'
				if sorted(tmp_q) == sorted(q1):
					iter_relevance[querys] = myinf
		rank = sorted(iter_relevance.items(), key=lambda score:score[1], reverse=True)
		rank_dict = {item[0]:item[1] for item in rank[:total_num]}
		#print('iteration ' + ': ', rank_dict)
		#with open('terms1.txt', 'a') as f1:
		#	with open('terms2.txt', 'a') as f2:
		#		for pair in tqdm(rank_dict):
		#			f1.write(pair[0] + '\n')
		#			f2.write(pair[1] + '\n')
		with open('3.txt', 'a') as f3:
			for pair in tqdm(rank_dict):
				f3.write(pair[0]+' '+pair[1] + '\n')



	def _simulateWalk(self, g_id, g_query, cur_query, steps, iter_single):

		if steps == 3:
			return

		for i in range(iter_single):
			mid_id = random.choice(g_query[cur_query])
			cnt1 = 0
			while mid_id not in g_id:
				mid_id = random.choice(g_query[cur_query])
				if cnt1 == 5:
					return
				cnt1 += 1
			rele_query = random.choice(g_id[mid_id])
			cnt2 = 0
			while rele_query == cur_query:
				rele_query = random.choice(g_id[mid_id])
				if cnt2 == 5:
					return
				cnt2 += 1
			self.relevance[tuple(sorted([cur_query, rele_query]))] += 1 / (steps ** 2)

			if rele_query in g_query:
				self._simulateWalk(g_id, g_query, rele_query, steps+1, iter_single)


	def ClassifyData_id(self):

		data = pd.read_csv(self.group_by_id_path)

		for _, row in data.iterrows():
			lengths = defaultdict(list)
			queries = row['_c1'].translate(self.tab).replace(' ', '').split('|')
			for query in queries:
				lengths[len(query)].append(query)
			for length, queries in lengths.items():
				queries = set(queries)
				if len(queries) == 1:
					continue
				with open('../data/' + str(length) + '_id.txt', 'a') as f:
					f.write(str(row['goods_id']) + ' ')
					for query in queries:
						f.write(query + ' ')
					f.write('\n')

	def ClassifyData_query(self):

		lines = open(self.group_by_query_path, 'r').readlines()[1:]

		for line in lines:
			line = line.strip().split(',')
			if len(line) != 2:
				continue
			#query = line[0].translate(self.tab).replace(' ', '')
			query = line[0]
			ids = line[1].split('|')
			with open('../data/' + str(len(query)) + '_query.txt', 'a') as f:
				f.write(query + ' ')
				for id_ in ids:
					f.write(id_ + ' ')
				f.write('\n')

	def __makeTable(self):

		punc = (string.punctuation + '，。《》【】、｜（）！？；\n\t：').replace('|', '')
		tab = str.maketrans(punc, ' '*len(punc))
		return tab

def func(i):
	print('processing', i)
	walk = Walker(i)
	walk.simulateWalk()
	print('process %d done' % i)

if __name__ == '__main__':
	process_list = []
	for i in range(3, 4):
		p = Process(target = func, args = (i,))
		p.start()
		process_list.append(p)
	for p in process_list:
		p.join()
		#walk = Walker(i)
		#walk.ClassifyData_id()
		#walk.ClassifyData_query()
		#print('done')
		#walk.simulateWalk()
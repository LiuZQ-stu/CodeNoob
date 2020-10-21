import pickle
import torch
import time
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('./')
sys.path.append('../')
from myutils.ner_evaluation.ner_eval import Evaluator

import torch.autograd as autograd
from model.gazlstm import GazLSTM as SeqModel
from transformers.tokenization_bert import BertTokenizer
from utils.functions import *


class Inference:
	def __init__(self, save_data_name, save_model_dir):

		self.num_layer = 4
		self.num_iter = 2
		self.labelcomment = ''
		self.resultfile = 'resultfile'
		self.lr = 0.0015
		self.use_biword = True
		self.use_char = True
		self.model_type = 'lstm'
		self.hidden_dim = 300
		self.use_count = True
		self.gpu = False
		self.data = self.load_data(save_data_name)
		self.model = self.load_model(save_model_dir)
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

	def load_data(self, save_data_name):

		with open(save_data_name, 'rb') as fp:
			data = pickle.load(fp)
		data.HP_num_layer = self.num_layer
		data.HP_iteration = self.num_iter
		data.label_comment = self.labelcomment
		data.result_file = self.resultfile
		data.HP_lr = self.lr
		data.use_bigram = self.use_biword
		data.HP_use_char = self.use_char
		data.model_type = self.model_type
		data.HP_hidden_dim = self.hidden_dim
		data.HP_use_count = self.use_count
		data.HP_gpu = self.gpu

		return data

	def load_model(self, model_dir):

		model = SeqModel(self.data)
		model.load_state_dict(torch.load(model_dir))

		return model

	def generate_instance_with_gaz(self, texts, labels):
		self.data.fix_alphabet()
		self.data.test_texts, self.data.test_Ids = self.read_instance_with_gaz(texts, labels, self.data.gaz, self.data.word_alphabet, self.data.biword_alphabet, self.data.biword_count, self.data.char_alphabet, self.data.gaz_alphabet, self.data.gaz_count, self.data.gaz_split,  self.data.label_alphabet, 100)

	def read_instance_with_gaz(self, texts, labels, gaz, word_alphabet, biword_alphabet, biword_count, char_alphabet, gaz_alphabet, gaz_count, gaz_split, label_alphabet, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):

		instance_texts = []
		instance_Ids = []
		words = []
		biwords = []
		chars = []
		word_Ids = []
		biword_Ids = []
		char_Ids = []
		if labels:
			labels = list(np.reshape(labels, [1, -1])[0])
			label_Ids = [label_alphabet.get_index(label) for label in labels]
		else:
			labels = ['O' for i in range(len(''.join(texts)))]
			label_Ids = [label_alphabet.get_index(label) for label in labels]
		for sentence in texts:
			for idx in range(len(sentence)):
				word = sentence[idx]
				word = normalize_word(word)
				if idx < len(sentence) - 1:
					biword = word + sentence[idx+1]
				else:
					biword = word + '-unknown-'
				biwords.append(biword)
				words.append(word)
				word_Ids.append(word_alphabet.get_index(word))
				biword_index = biword_alphabet.get_index(biword)
				biword_Ids.append(biword_index)
				char_list = []
				char_Id = []
				for char in word:
					char_list.append(char)
				if char_padding_size > 0:
					char_number = len(char_list)
					if char_number < char_padding_size:
						char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
					assert(len(char_list) == char_padding_size)
				else:
					### not padding
					pass
				for char in char_list:
					char_Id.append(char_alphabet.get_index(char))
				chars.append(char_list)
				char_Ids.append(char_Id)
			if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words)>0):
				gaz_Ids = []
				layergazmasks = []
				gazchar_masks = []
				w_length = len(words)

				gazs = [ [[] for i in range(4)] for _ in range(w_length)]  # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
				gazs_count = [ [[] for i in range(4)] for _ in range(w_length)]

				gaz_char_Id = [ [[] for i in range(4)] for _ in range(w_length)]  ## gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[[w1c1,w1c2,...],[],...]

				max_gazlist = 0
				max_gazcharlen = 0

				for idx in range(w_length):

					matched_list = gaz.enumerateMatchList(words[idx:])
					matched_length = [len(a) for a in matched_list]
					matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]

					if matched_length:
						max_gazcharlen = max(max(matched_length),max_gazcharlen)

					for w in range(len(matched_Id)):
						gaz_chars = []
						g = matched_list[w]
						for c in g:
							gaz_chars.append(word_alphabet.get_index(c))

						if matched_length[w] == 1:  ## Single
							gazs[idx][3].append(matched_Id[w])
							gazs_count[idx][3].append(1)
							gaz_char_Id[idx][3].append(gaz_chars)
						else:
							gazs[idx][0].append(matched_Id[w])   ## Begin
							gazs_count[idx][0].append(gaz_count[matched_Id[w]])
							gaz_char_Id[idx][0].append(gaz_chars)
							wlen = matched_length[w]
							gazs[idx+wlen-1][2].append(matched_Id[w])  ## End
							gazs_count[idx+wlen-1][2].append(gaz_count[matched_Id[w]])
							gaz_char_Id[idx+wlen-1][2].append(gaz_chars)
							for l in range(wlen-2):
								gazs[idx+l+1][1].append(matched_Id[w])  ## Middle
								gazs_count[idx+l+1][1].append(gaz_count[matched_Id[w]])
								gaz_char_Id[idx+l+1][1].append(gaz_chars)



					for label in range(4):
						if not gazs[idx][label]:
							gazs[idx][label].append(0)
							gazs_count[idx][label].append(1)
							gaz_char_Id[idx][label].append([0])

						max_gazlist = max(len(gazs[idx][label]),max_gazlist)

					matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]  #词号
					if matched_Id:
						gaz_Ids.append([matched_Id, matched_length])
					else:
						gaz_Ids.append([])

				t1 = time.time()
				## batch_size = 1
				for idx in range(w_length):
					gazmask = []
					gazcharmask = []

					for label in range(4):
						label_len = len(gazs[idx][label])
						count_set = set(gazs_count[idx][label])
						if len(count_set) == 1 and 0 in count_set:
							gazs_count[idx][label] = [1]*label_len

						mask = label_len*[0]
						mask += (max_gazlist-label_len)*[1]

						gazs[idx][label] += (max_gazlist-label_len)*[0]  ## padding
						gazs_count[idx][label] += (max_gazlist-label_len)*[0]  ## padding

						char_mask = []
						for g in range(len(gaz_char_Id[idx][label])):
							glen = len(gaz_char_Id[idx][label][g])
							charmask = glen*[0]
							charmask += (max_gazcharlen-glen) * [1]
							char_mask.append(charmask)
							gaz_char_Id[idx][label][g] += (max_gazcharlen-glen) * [0]
						gaz_char_Id[idx][label] += (max_gazlist-label_len)*[[0 for i in range(max_gazcharlen)]]
						char_mask += (max_gazlist-label_len)*[[1 for i in range(max_gazcharlen)]]

						gazmask.append(mask)
						gazcharmask.append(char_mask)
					layergazmasks.append(gazmask)
					gazchar_masks.append(gazcharmask)

				texts = ['[CLS]'] + words + ['[SEP]']
				bert_text_ids = self.tokenizer.convert_tokens_to_ids(texts)


				instance_texts.append([words, biwords, chars, gazs, labels])
				instance_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids, gazs, gazs_count, gaz_char_Id, layergazmasks,gazchar_masks, bert_text_ids])


		return instance_texts, instance_Ids

	def evaluate(self, data, model):

		instances = data.test_Ids
		pred_results = []
		gold_results = []
		## set model in eval model
		model.eval()
		batch_size = 1
		train_num = len(instances)
		total_batch = train_num // batch_size + 1
		for batch_id in range(total_batch):
			with torch.no_grad():
				start = batch_id * batch_size
				end = (batch_id + 1) * batch_size
				if end > train_num:
					end = train_num
				instance = instances[start:end]
				if not instance:
					continue
				gaz_list, batch_word, batch_biword, batch_wordlen, batch_label, layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask, batch_bert, bert_mask = batchify_with_label(
					instance, data.HP_gpu, data.HP_num_layer, True)
				tag_seq, gaz_match = model(gaz_list, batch_word, batch_biword, batch_wordlen, layer_gaz, gaz_count,
										   gaz_chars, gaz_mask, gazchar_mask, mask, batch_bert, bert_mask)


				pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet)
				pred_results += pred_label
				gold_results += gold_label

		return pred_results

	def Infer(self, texts, labels = None):

		start_time = time.time()
		if isinstance(texts, str):
			texts = [texts]
		try:
			self.generate_instance_with_gaz(texts, labels)
		except:
			return [], 0
		results = self.evaluate(self.data, self.model)
		end_time = time.time()
		time_cost = end_time - start_time
		#print(("%s: time:%.5fs"%('Infer', time_cost)))
		if time_cost > 0.05:
			print(len(texts[0]))
		return results, time_cost

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet):

	batch_size = gold_variable.size(0)
	seq_len = gold_variable.size(1)
	mask = mask_variable.cpu().data.numpy()
	pred_tag = pred_variable.cpu().data.numpy()
	gold_tag = gold_variable.cpu().data.numpy()
	batch_size = mask.shape[0]
	pred_label = []
	gold_label = []
	for idx in range(batch_size):
		pred = [label_alphabet.get_instance(int(pred_tag[idx][idy])) for idy in range(seq_len) if mask[idx][idy] != 0]
		gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]

		assert(len(pred)==len(gold))
		pred_label.append(pred)
		gold_label.append(gold)

	return pred_label, gold_label

def batchify_with_label(input_batch_list, gpu, num_layer, volatile_flag=False):

	batch_size = len(input_batch_list)
	words = [sent[0] for sent in input_batch_list]
	biwords = [sent[1] for sent in input_batch_list]
	gazs = [sent[3] for sent in input_batch_list]
	labels = [sent[4] for sent in input_batch_list]
	layer_gazs = [sent[5] for sent in input_batch_list]
	gaz_count = [sent[6] for sent in input_batch_list]
	gaz_chars = [sent[7] for sent in input_batch_list]
	gaz_mask = [sent[8] for sent in input_batch_list]
	gazchar_mask = [sent[9] for sent in input_batch_list]
	bert_ids = [sent[10] for sent in input_batch_list]

	word_seq_lengths = torch.LongTensor(list(map(len, words)))
	max_seq_len = word_seq_lengths.max()
	word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
	biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
	label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
	mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
	### bert seq tensor
	bert_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len+2))).long()
	bert_mask = autograd.Variable(torch.zeros((batch_size, max_seq_len+2))).long()

	gaz_num = [len(layer_gazs[i][0][0]) for i in range(batch_size)]
	max_gaz_num = max(gaz_num)
	layer_gaz_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num).long()
	gaz_count_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num).float()
	gaz_len = [len(gaz_chars[i][0][0][0]) for i in range(batch_size)]
	max_gaz_len = max(gaz_len)
	gaz_chars_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num, max_gaz_len).long()
	gaz_mask_tensor = torch.ones(batch_size, max_seq_len, 4, max_gaz_num).byte()
	gazchar_mask_tensor = torch.ones(batch_size, max_seq_len, 4, max_gaz_num, max_gaz_len).byte()

	for b, (seq, bert_id, biseq, label, seqlen, layergaz, gazmask, gazcount, gazchar, gazchar_mask, gaznum, gazlen) in enumerate(zip(words, bert_ids, biwords, labels, word_seq_lengths, layer_gazs, gaz_mask, gaz_count, gaz_chars, gazchar_mask, gaz_num, gaz_len)):

		word_seq_tensor[b, :seqlen] = torch.LongTensor(seq)
		biword_seq_tensor[b, :seqlen] = torch.LongTensor(biseq)
		label_seq_tensor[b, :seqlen] = torch.LongTensor(label)
		layer_gaz_tensor[b, :seqlen, :, :gaznum] = torch.LongTensor(layergaz)
		mask[b, :seqlen] = torch.Tensor([1]*int(seqlen))
		bert_mask[b, :seqlen+2] = torch.LongTensor([1]*int(seqlen+2))
		gaz_mask_tensor[b, :seqlen, :, :gaznum] = torch.ByteTensor(gazmask)
		gaz_count_tensor[b, :seqlen, :, :gaznum] = torch.FloatTensor(gazcount)
		gaz_count_tensor[b, seqlen:] = 1
		gaz_chars_tensor[b, :seqlen, :, :gaznum, :gazlen] = torch.LongTensor(gazchar)
		gazchar_mask_tensor[b, :seqlen, :, :gaznum, :gazlen] = torch.ByteTensor(gazchar_mask)

		##bert
		bert_seq_tensor[b, :seqlen+2] = torch.LongTensor(bert_id)


	if gpu:
		word_seq_tensor = word_seq_tensor.cuda()
		biword_seq_tensor = biword_seq_tensor.cuda()
		word_seq_lengths = word_seq_lengths.cuda()
		label_seq_tensor = label_seq_tensor.cuda()
		layer_gaz_tensor = layer_gaz_tensor.cuda()
		gaz_chars_tensor = gaz_chars_tensor.cuda()
		gaz_mask_tensor = gaz_mask_tensor.cuda()
		gazchar_mask_tensor = gazchar_mask_tensor.cuda()
		gaz_count_tensor = gaz_count_tensor.cuda()
		mask = mask.cuda()
		bert_seq_tensor = bert_seq_tensor.cuda()
		bert_mask = bert_mask.cuda()

	# print(bert_seq_tensor.type())
	return gazs, word_seq_tensor, biword_seq_tensor, word_seq_lengths, label_seq_tensor, layer_gaz_tensor, gaz_count_tensor,gaz_chars_tensor, gaz_mask_tensor, gazchar_mask_tensor, mask, bert_seq_tensor, bert_mask

if __name__ == '__main__':


	'''INFER = Inference('data/data_goods.dset', 'save_model/goods_train_hd300')
	data = open('data/goods_test.char').readlines()

	text = ''
	label = []
	pred = []
	labels = []
	total_time = []
	with open('res','w') as f:
		for idx, line in tqdm(enumerate(data)):
			if 680000 < idx < 690000:
				continue
			line = line.strip()
			if len(line) > 2:
				line = line.split()
				text += line[0]
				label.append(line[1])
			else:
				res, time_cost = INFER.Infer(text)
				if not res:
					text = ''
					label = []
					continue
				res = res[0]
				if len(res) == len(label):
					pred.append(res)
					labels.append(label)
				for a, b, c in zip(text, res, label):
					f.write(' '.join([a,b,c]) + '\n')
				text = ''
				label = []
				total_time.append(time_cost)
	t1 = time.time()
	evaluator = Evaluator(pred, labels, ['people', 'region', 'element', 'adjective', 'style', 'season', 'material', 'function'])
	results, results_agg = evaluator.evaluate()
	t2 = time.time()
	print(results['strict'])
	for key in results_agg:
		print(key, results_agg[key]['strict'])
	print('average time cost:', sum(total_time)/len(total_time))
	print('total time cost:', sum(total_time))
	print('evaluation time:', t2-t1)'''

	INFER = Inference('data/data_goods.dset', 'save_model/goods_train_hd300')
	print(INFER.Infer('真皮男鞋'))





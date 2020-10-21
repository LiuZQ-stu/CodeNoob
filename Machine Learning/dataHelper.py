import os
import tqdm

class DataMaker:
	def __init__(self):

		self.num_partition = 20
		self.data_file = '/home/userroot/zhaoqi/project1/LexiconAugmentedNER-master/data'
		self.train_target = 'goods'

	def Partition(self, length):

		portion = length // self.num_partition
		scaler = [portion * i for i in range(1, self.num_partition)]

		return [1] + scaler + [length]

	def DataLength(self):

		data_path = self.train_target + '_train_full.char'
		data_path = os.path.join(self.data_file, data_path)
		data = open(data_path, 'rb').readlines()

		return len(data)

	def MakeTrainData(self, boundary):  #boundary: (left, right)
		
		data_path = self.train_target + '_train_full.char'
		data_path = os.path.join(self.data_file, data_path)
		target_path = self.train_target + '_train.char'
		target_path = os.path.join(self.data_file, target_path)

		data = open(data_path, 'rb').readlines()
		with open(target_path, 'w', encoding='utf8') as f:
			for rdx in range(boundary[0], boundary[1]):
				f.write(data[rdx].decode('utf-8'))
			f.write('/n')



	def MakeDevAndTest(self, index1, index2):

		data_path = [self.train_target + '_dev_full.char', self.train_target + '_test_full.char']
		data_path = [os.path.join(self.data_file, file) for file in data_path]
		target_path = [self.train_target + '_dev.char', self.train_target + '_test.char']
		target_path = [os.path.join(self.data_file, file) for file in target_path]

		for data, file in zip(data_path, target_path):
			inlines = open(data, 'rb').readlines()
			length = len(inlines)
			scaler = self.Partition(length)
			boundary = [scaler[index1], scaler[index2]]
			with open(file, 'w') as f:
				for i in range(boundary[0], boundary[1]):
					f.write(inlines[i].decode('utf-8'))
				f.write('/n')

	def test(self):

		self.MakeDevAndTest()
		length = self.DataLength()
		scaler = self.Partition(length)

		for idx in range(len(scaler)-1):
			boundary = (scaler[idx], scaler[idx+1])
			self.MakeTrainData(boundary)


if __name__ == '__main__':


	datamaker = DataMaker()
	datamaker.MakeDevAndTest()
	datamaker.MakeTrainData((1, datamaker.DataLength()))
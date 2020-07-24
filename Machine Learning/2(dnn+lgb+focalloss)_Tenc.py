import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
AUTOTUNE = tf.data.experimental.AUTOTUNE
from sklearn.model_selection import train_test_split



class Model:
	def __init__(self, user_path, userVecPath):

		self.user = pd.read_csv(user_path, header=0, encoding='utf8')
		self.userVec = np.load(userVecPath, allow_pickle=True).item()
		self.trainX, self.testX, self.label_age, self.label_gender = self.getData()
		self.num_train = 900000
		self.num_test = 1000000

	def getData(self):

		self.num_train = 900000
		self.num_test = 1000000
		
		label_age = np.array([ [0] * 10 for _ in range(self.num_train)])
		label_gender = np.array([ 0 for _ in range(self.num_train)])
		trainX, testX = [], []

		for idx, row in self.user.iterrows():
			label_age[idx][row['age'] - 1] = 1
			label_gender[idx] = row['gender'] - 1

		for trn_user in range(1, self.num_train + 1):
			trainX.append(self.userVec[str(trn_user)])
		for tst_user in range(3000001, 4000001):
			testX.append(self.userVec[str(tst_user)])

		trainX = np.array(trainX)
		testX = np.array(testX)
		print('the shape of trainX is:' , np.shape(trainX))
		print('the shape of testX is:' , np.shape(testX))
		print('the first item of label_age is:', label_age[0])
		print('the first item of label_gender is:', label_gender[0])

		return trainX, testX, label_age, label_gender

	def rearge_labels(self, age_label):

		new_label7 = [[0] * 7 for _ in range(len(age_label))]

		for idx, label in enumerate(age_label):
			new_label7[idx] = label[:7]
			if 1 not in new_label7[idx]:
				new_label7[idx][-1] = 1

		return new_label7

	def sex_split(self):

		trainX_man = []
		trainY_man = []
		trainX_woman = []
		trainY_woman = []

		for n in range(self.num_train):
			if not self.label_gender[n]:
				trainX_man.append(self.trainX[n])
				trainY_man.append(self.label_age[n])
			else:
				trainX_woman.append(self.trainX[n])
				trainY_woman.append(self.label_age[n])

		return trainX_man, trainY_man, trainX_woman, trainY_woman

	def train(self):

		trainX, valX, trainY, valY = train_test_split(
				self.trainX,
				self.label_gender,
				test_size = 0.05,
				random_state = 1,
				stratify = self.label_gender
				)

		param = { 
			'boosting_type': 'gbdt',
			'objective': 'binary', 
			'metric': {'binary_logloss', 'auc'},
			'num_leaves': 60,
			'max_depth': 6,
			'min_data_in_leaf': 400,
			'learning_rate': 0.03,
			'feature_fraction': 0.9,
			'bagging_fraction': 0.9,
			'bagging_freq': 4,
			'lambda_l1': 0.2,
			'lambda_l2': 0.2,
			'verbose': -1,
			'min_gain_to_split': 0.2,
			'num_threads': 4,
			'random_state': 2019,
			'is_unbalance': True
		}

		trn_ds = lgb.Dataset(trainX, trainY)
		val_ds = lgb.Dataset(valX, valY, reference=trn_ds)

		clf = lgb.train(param,
						trn_ds,
						num_boost_round=3000,
						valid_sets=val_ds,
						early_stopping_rounds=150,
						)

		pred_gender = clf.predict(self.testX, num_iteration=clf.best_iteration)

		threshold = 0.5
		test_man = []
		test_woman = []
		man_list = []
		woman_list = []
		for idx, res in enumerate(pred_gender):
			if res > threshold:
				pred_gender[idx] = 2
				test_woman.append(self.testX[idx])
				woman_list.append(idx + 1)
			else:
				pred_gender[idx] = 1
				test_man.append(self.testX[idx])
				man_list.append(idx + 1)

		del self.testX

		test_man = np.array(test_man)
		test_woman = np.array(test_woman)

		trainX_man, trainY_man, trainX_woman, trainY_woman = self.sex_split()

		del self.trainX

		cls7_label_man = self.rearge_labels(trainY_man)
		cls7_label_woman = self.rearge_labels(trainY_woman)

		print("the shape of trainX_man is:", np.shape(trainX_man))
		print("the shape of cls7_label_man is:", np.shape(cls7_label_man))
		print("the shape of trainX_woman is:", np.shape(trainX_woman))
		print("the shape of cls7_label_woman is:", np.shape(cls7_label_woman))

		man_pred_7age = self.trainForOneGender(trainX_man, cls7_label_man, test_man, 7)
		woman_pred_7age = self.trainForOneGender(trainX_woman, cls7_label_womans, test_woman, 7)

		cls4_trainX_man = []
		cls4_trainY_man = []
		for idx, item in enumerate(trainY_man):
			item = list(item)
			if trainY_man.index(1) >= 6:
				cls4_trainX_man.append(trainX_man[idx])
				cls4_trainY_man.append(item[-4:])
		
		cls4_trainX_woman = []
		cls4_trainY_woman = []
		for idx, item in enumerate(trainY_woman):
			item = list(item)
			if trainY_man.index(1) >= 6:
				cls4_trainX_woman.append(trainX_woman[idx])
				cls4_trainY_woman.append(item[-4:])

		test_man_4 = []
		test_woman_4 = []
		for idx, item in enumerate(man_pred_7age):
			if item == 7:
				test_man_4.append(test_man[idx])
		for idx, item in enumerate(woman_pred_7age):
			if item == 7:
				test_woman_4.append(test_woman[idx])

		man_pred_4age = self.trainForOneGender(cls4_trainX_man,
										cls4_trainY_man, test_man_4, 4)
		woman_pred_4age = self.trainForOneGender(cls4_trainX_woman,
										cls4_trainY_woman, test_woman_4, 4)

		del trainX_woman, trainX_man

		ptr = 0
		for idx, pred in enumerate(man_pred_7age):
			if pred == 7:
				man_pred_7age[idx] = man_pred_4age[ptr] + 6
				ptr += 1
		ptr = 0
		for idx, pred in enumerate(woman_pred_7age):
			if pred == 7:
				woman_pred_7age[idx] = woman_pred_4age[ptr] + 6
				ptr += 1

		return man_pred_7age + woman_pred_7age, man_list, woman_list

	def trainForOneGender(self, trainX, trainY, testX, num_class):

		trainX = np.array(trainX)
		trainY = np.array(trainY)
		testX = np.array(testX)

		trainX, valX, trainY, valY = train_test_split(
				trainX,
				trainY,
				test_size = 0.05,
				random_state = 1,
				stratify = trainY
				)

		train_ds = tf.data.Dataset.from_tensor_slices((trainX, trainY))

		BATCH_SIZE = 256
		train_ds = train_ds.apply(
					tf.data.experimental.shuffle_and_repeat(buffer_size=len(trainY)))
		train_ds = train_ds.batch(BATCH_SIZE)
		train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
		val_ds = tf.data.Dataset.from_tensor_slices((valX, valY)).batch(BATCH_SIZE)
		test_ds = tf.data.Dataset.from_tensor_slices(testX).batch(BATCH_SIZE)

		model = keras.Sequential([
			keras.layers.Dense(256, activation='sigmoid', input_dim=150),
			keras.layers.Dropout(0.4),
			keras.layers.Dense(256, activation='sigmoid'),
			keras.layers.Dense(num_class, activation='softmax'),
			])
		
		alpha = {'7':[[0.97],[0.85],[0.8],[0.84],[0.87],[0.90],[0.76]],
				'4':[[0.47],[0.756],[0.85],[0.92]]}
		model.compile(optimizer='adam',
			loss=[self._multi_category_focal_loss1(alpha=alpha[str(num_class)], gamma=2)],
			metrics=['accuracy'])
		model.fit(train_ds, epochs=50, steps_per_epoch=len(trainY)//BATCH_SIZE)
		model.evaluate(val_ds, verbose=2)

		prediction = model.predict(test_ds)
		pred_age = [ 0 for _ in range(len(testX))]
		for idx, res in enumerate(prediction):
			pred_age[idx] = int(np.argmax(res)) + 1

		return pred_age


	def _multi_category_focal_loss1(self, alpha, gamma=2.0):

		epsilon = 1.e-7
		alpha = tf.constant(alpha, dtype=tf.float32)
		gamma = float(gamma)

		def multi_category_focal_loss1_fixed(y_true, y_pred):

			y_true = tf.cast(y_true, tf.float32)
			y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
			y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
			ce = -tf.math.log(y_t)
			weight = tf.pow(tf.subtract(1., y_t), gamma)
			fl = tf.matmul(tf.multiply(weight, ce), alpha)
			loss = tf.reduce_mean(fl)

			return loss

		return multi_category_focal_loss1_fixed

	def saveAns(self, pred_age, man_list, woman_list):

		user_list = man_list + woman_list
		pred_gender = np.append(np.zeros(len(man_list)), np.ones(len(woman_list)))
		pred_gender = list(map(int, pred_gender + np.ones(self.num_test)))
		ans = pd.DataFrame({'user_id': user_list,
							'predicted_age': pred_age,
							'predicted_gender': pred_gender
							})
		ans.to_csv('submission.csv', sep=',', index=False, encoding='utf8')



if __name__ == '__main__':

	user_path = ''
	userVecPath = ''

	model = Model(user_path, userVecPath)
	pred_age, man_list, woman_list = model.train()
	model.saveAns(pred_age, man_list, woman_list)
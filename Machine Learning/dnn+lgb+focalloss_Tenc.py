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

	def train(self):

		trainX, valX, trainY, val_label_age = train_test_split(
				self.trainX,
				self.label_age,
				test_size = 0.05,
				random_state = 1,
				stratify = self.label_age
				)

		age_train_ds = tf.data.Dataset.from_tensor_slices((trainX, trainY))

		BATCH_SIZE = 256
		age_train_ds = age_train_ds.apply(
					tf.data.experimental.shuffle_and_repeat(buffer_size=self.num_train))
		age_train_ds = age_train_ds.batch(BATCH_SIZE)
		age_train_ds = age_train_ds.prefetch(buffer_size=AUTOTUNE)
		age_val_ds = tf.data.Dataset.from_tensor_slices((valX, val_label_age)).batch(BATCH_SIZE)
		test_ds = tf.data.Dataset.from_tensor_slices(self.testX).batch(BATCH_SIZE)

		model = keras.Sequential([
			keras.layers.Dense(256, activation='sigmoid', input_dim=150),
			keras.layers.Dropout(0.4),
			keras.layers.Dense(256, activation='sigmoid'),
			keras.layers.Dense(10, activation='softmax'),
			])
		
		alpha = [[0.96],[0.83],[0.77],[0.83],[0.85],[0.88],[0.92],[0.96],[0.978],[0.987]]
		model.compile(optimizer='adam',
			loss=[self._multi_category_focal_loss1(alpha=alpha, gamma=2)],
			metrics=['accuracy'])
		model.fit(age_train_ds, epochs=50, steps_per_epoch=self.num_train//BATCH_SIZE)
		model.evaluate(age_val_ds, verbose=2)

		prediction = model.predict(test_ds)
		pred_age = [ 0 for _ in range(self.num_test)]
		for idx, res in enumerate(prediction):
			pred_age[idx] = int(np.argmax(res)) + 1


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
			'num_leaves': 100,
			'max_depth': 7,
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
		for idx, res in enumerate(pred_gender):
			pred_gender[idx] = 2 if res > threshold else 1

		return pred_age, pred_gender

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

	def saveAns(self, pred_age, pred_gender):

		user_list = np.array([num for num in range(3000001, 4000001)])
		ans = pd.DataFrame({'user_id': user_list,
							'predicted_age': pred_age,
							'predicted_gender': pred_gender
							})
		ans.to_csv('submission.csv', sep=',', index=False, encoding='utf8')



if __name__ == '__main__':

	user_path = ''
	userVecPath = ''

	model = Model(user_path, userVecPath)
	pred_age, pred_gender = model.train()
	model.saveAns(pred_age, pred_gender)
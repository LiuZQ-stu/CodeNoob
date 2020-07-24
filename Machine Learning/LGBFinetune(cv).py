import numpy as np
import pandas as pd
import os
import tensorflow as tf
import lightgbm as lgb
from sklearn.model_selection import KFold
from tensorflow import keras
AUTOTUNE = tf.data.experimental.AUTOTUNE



def train_data_process(user_path, embeddings_path):

	train_num = 700000
	test_num = 200000
	user = pd.read_csv(user_path,header=0,encoding='utf8')
	y_train = [[0] * 10 for _ in range(train_num)]
	y_test = [[0] * 10 for _ in range(test_num)]
	embeddings = np.load(embeddings_path, allow_pickle=True)
	x_train, x_test, user_list = [], [], []
	for user_id in range(len(embeddings)):
		temp = user[user.user_id==(user_id+1)]
		if user_id < train_num:
			y_train[user_id][temp.iat[0,1] - 1] = 1
			x_train.append(list(embeddings[user_id]))
		else:
			y_test[user_id-train_num][temp.iat[0,1] - 1] = 1
			x_test.append(list(embeddings[user_id]))
			user_list.append(user_id+1)

	return x_train, y_train, x_test, y_test, user_list

def first_clf_label(label):

	new_label = [ [0] * 6 for _ in range(len(label))]
	for idx, target in enumerate(label):
		if target.index(1) >= 6:
			new_label[idx][0] = 1
		else:
			new_label[idx][target.index(1)] = 1

	return new_label

def saveMyAns(res1, res2, sub_user_list, user_list):

	sub_user_dict = {}
	for i in range(len(res2)):
		sub_user_dict[str(sub_user_list[i])] = res2[i]
	pred_age = []
	for idx, pred in enumerate(res1):
		n = pred.index(max(pred))
		if n != 0:
			pred_age.append(n + 1)
		else:
			temp = sub_user_dict[str(idx + 700001)]
			if not temp.index(max(temp)):
				pred_age.append(1)
			else:
				pred_age.append(temp.index(max(temp)) + 6)
	df = pd.DataFrame({'user_id':user_list,
					'age': pred_age})
	df.to_csv('result.csv', sep=',', index=False, encoding='utf8')

def train1(x_train, y_train, x_test, y_test1):

	lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
	lgb_eval = lgb.Dataset(X_test, y_test1, reference=lgb_train,free_raw_data=False)
	
	params = {
			'boosting_type': 'gbdt',
			'objective': 'multiclass',
			'metric': 'multi_error',
			'num_class': 6,
			}

	min_merror = float('Inf')
	best_params = {}

	print("调参1：提高准确率")
	for num_leaves in range(20,200,5):
		for max_depth in range(3,8,1):
			params['num_leaves'] = num_leaves
			params['max_depth'] = max_depth

			cv_results = lgb.cv(
								params,
								lgb_train,
								seed=2018,
								nfold=3,
								early_stopping_rounds=10,
								verbose_eval=True
								)

			mean_merror = pd.Series(cv_results['multi_error-mean']).min()
			boost_rounds = pd.Series(cv_results['multi_error-mean']).argmin()

			if mean_merror < min_merror:
				min_merror = mean_merror
				best_params['num_leaves'] = num_leaves
				best_params['max_depth'] = max_depth

	params['num_leaves'] = best_params['num_leaves']
	params['max_depth'] = best_params['max_depth']

	print("调参2：降低过拟合")
	for max_bin in range(1,255,5):
		for min_data_in_leaf in range(10,200,5):
			params['max_bin'] = max_bin
			params['min_data_in_leaf'] = min_data_in_leaf

			cv_results = lgb.cv(
								params,
								lgb_train,
								seed=42,
								nfold=3,
								early_stopping_rounds=3,
								verbose_eval=True
								)

			mean_merror = pd.Series(cv_results['multi_error-mean']).min()
			boost_rounds = pd.Series(cv_results['multi_error-mean']).argmin()

			if mean_merror < min_merror:
				min_merror = mean_merror
				best_params['max_bin'] = max_bin
				best_params['min_data_in_leaf'] = min_data_in_leaf

	params['min_data_in_leaf'] = best_params['min_data_in_leaf']
	params['max_bin'] = best_params['max_bin']

	print("调参3：降低过拟合")
	for feature_fraction in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
		for bagging_fraction in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
			for bagging_freq in range(0,50,5):
				params['feature_fraction'] = feature_fraction
				params['bagging_fraction'] = bagging_fraction
				params['bagging_freq'] = bagging_freq

				cv_results = lgb.cv(
									params,
									lgb_train,
									seed=42,
									nfold=3,
									early_stopping_rounds=3,
									verbose_eval=True
									)

				mean_merror = pd.Series(cv_results['multi_error-mean']).min()
				boost_rounds = pd.Series(cv_results['multi_error-mean']).argmin()

				if mean_merror < min_merror:
					min_merror = mean_merror
					best_params['feature_fraction'] = feature_fraction
					best_params['bagging_fraction'] = bagging_fraction
					best_params['bagging_freq'] = bagging_freq

	params['feature_fraction'] = best_params['feature_fraction']
	params['bagging_fraction'] = best_params['bagging_fraction']
	params['bagging_freq'] = best_params['bagging_freq']

	print("调参4：降低过拟合")
	for lambda_l1 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
		for lambda_l2 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
			for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
				params['lambda_l1'] = lambda_l1
				params['lambda_l2'] = lambda_l2
				params['min_split_gain'] = min_split_gain

				cv_results = lgb.cv(
									params,
									lgb_train,
									seed=42,
									nfold=3,
									early_stopping_rounds=3,
									verbose_eval=True
									)

				mean_merror = pd.Series(cv_results['multi_error-mean']).min()
				boost_rounds = pd.Series(cv_results['multi_error-mean']).argmin()

				if mean_merror < min_merror:
					min_merror = mean_merror
					best_params['lambda_l1'] = lambda_l1
					best_params['lambda_l2'] = lambda_l2
					best_params['min_split_gain'] = min_split_gain

	params['lambda_l1'] = best_params['lambda_l1']
	params['lambda_l2'] = best_params['lambda_l2']
	params['min_split_gain'] = best_params['min_split_gain']


	print(best_params)

	params['learning_rate']=0.01
	lgb.train(
			params,
			lgb_train,
			valid_sets=lgb_eval,
			num_boost_round=200,
			early_stopping_rounds=50
			)

	predictions = list(predictions)
	res = list(map(list, predictions))

	return res

def train2(sub_train_ds, sub_testX, sub_testY, sub_count, BATCH_SIZE = 512):

	

	model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		#loss='binary_crossentropy',
		metrics=['accuracy'], run_eagerly = True)
	model.fit(sub_train_ds, epochs=30, steps_per_epoch=sub_count//BATCH_SIZE,
		callbacks = callbacks)
	#model.evaluate(val_ds_1, verbose=2)
	res = list(model.predict(sub_test_ds))
	res = list(map(list, res))

	return res

def trainDataSelect(x_train, y_train):

	sub_trainX = []
	sub_trainY = []
	for idx, label in enumerate(y_train):
		if (label.index(1) >= 6) or not label.index(1):
			sub_trainX.append(x_train[idx])
			sub_trainY.append(label[:1] + label[-4:])
	sub_train_ds = tf.data.Dataset.from_tensor_slices((sub_trainX, sub_trainY))
	
	return sub_train_ds, len(sub_trainY)

def resultDataSelect(res1, x_test, y_test, user_list):

	sub_testX, sub_testY, sub_user_list = [], [], []
	for idx, pred in enumerate(res1):
		if pred[0] == max(pred):
			sub_testX.append(x_test[idx])
			sub_testY.append(pred)
			sub_user_list.append(user_list[idx])

	return sub_testX, sub_testY, sub_user_list





if __name__ == "__main__":

	file_path = ''
	user_path = os.path.join(file_path, 'user.csv')
	embeddings_path  = 'userVec128_arg.npy'

	print("train data processing")
	x_train, y_train, x_test, y_test, user_list = train_data_process(user_path, embeddings_path)
	y_train1 = first_clf_label(y_train)
	y_test1 = first_clf_label(y_test)
	print("slicing train ds")
	res1 = train1(x_train, y_train, x_test, y_test1) #6dim
	
	print("processing result 1")
	sub_testX, sub_testY, sub_user_list = resultDataSelect(res1, x_test, y_test, user_list)
	sub_train_ds, sub_count = trainDataSelect(x_train, y_train)
	res2 = train2(sub_train_ds, sub_testX, sub_testY, sub_count) #5dim
	print("saving")
	saveMyAns(res1, res2, sub_user_list, user_list)
	#repeat()
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score


sys.path.append('/home/aistudio/external-libraries')



def train_data_process(user_path, embeddings_path):

	user = pd.read_csv(user_path,header=0,encoding='utf8')
	global user_num
	user_num = len(user)
	age_label = [ 0 for _ in range(user_num)]
	gender_label = [ 0 for _ in range(user_num)]
	embeddings = np.load(embeddings_path).item()
	tensor_of_user = [[0]*len(embeddings['30920']) for _ in range(user_num)]
	i = 0
	for key in embeddings:
		temp = user[user.user_id==int(key)]
		age_label[i] = temp.iat[0,1] - 1
		gender_label[i] = temp.iat[0,2] - 1
		tensor_of_user[i] = embeddings[key]
		i += 1		

	return tensor_of_user, age_label, gender_label

def RandomForest(tensor_of_user, age_label, gender_label):
	
	k_range = range(30,100)
	age_cv_scores, gender_cv_scores = [], []
	for n in k_range:
		clf_age = RandomForestClassifier(n)
		clf_gender = RandomForestClassifier(n)
		score_age = cross_val_score(clf_age, tensor_of_user, age_label, cv=10, scoring='accuracy')
		score_gender = cross_val_score(clf_gender, tensor_of_user, gender_label, cv=10, scoring='accuracy')
		age_cv_scores.append(score_age)
		gender_cv_scores.append(score_gender)

	return age_cv_scores, gender_cv_scores

if __name__ == "__main__":

	

	file_path = 'data/data35459/'
	user_path = os.path.join(file_path, 'user.csv')
	embeddings_path  = 'data/data36688/userVec128.npy'

	print("preparing database...")
	tensor_of_user, age_label, gender_label = train_data_process(user_path, embeddings_path)
	age_cv_scores, gender_cv_scores = RandomForest(tensor_of_user, age_label, gender_label)
	print('done!')

	plt.plot(k_range, age_cv_scores)
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.title('age')
	plt.show()
	plt.plot(k_range, gender_cv_scores)
	plt.xlabel('K')
	plt.ylabel('Accuracy')
	plt.title('gender')
	plt.show()

	x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(tensor_of_user,
														label_age, test_size=0.25)
	x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(tensor_of_user,
														label_age, test_size=0.25)

	best_k1 = age_cv_scores.index(max(age_cv_scores)) + 30
	best_k2 = gender_cv_scores.index(max(gender_cv_scores)) + 30
	best_clf_age = RandomForestClassifier(best_k1)
	best_clf_gender = RandomForestClassifier(best_k2)
	best_clf_age.fit(x_train_age, y_train_age)
	best_clf_gender.fit(x_train_gender, y_train_gender)
	print(best_clf_age.score(x_test_age, y_test_age))
	print(best_clf_gender.score(x_test_gender, y_tesnt_gender))
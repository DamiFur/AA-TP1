# Aprendizaje Automatico

# set this variable to true for competition
BEGIN_COMPETITION = True

import sys
import os.path
import time

import json
import numpy as np
import pandas as pd

import email
import string
import re
from html.parser import HTMLParser

from collections import OrderedDict

import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # Bag of words
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel, chi2, SelectKBest
from sklearn.svm import LinearSVC

import pickle # model persistance
from matplotlib import pyplot as plt

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        if self.name:
            print(self.name),
        self.tstart = time.time()
    def __exit__(self, type, value, traceback):
        print('Elapsed: %.2f seconds' % (time.time() - self.tstart))

class MLStripper(HTMLParser):
	def __init__(self):
		super().__init__()
		self.reset()
		self.fed = []

	def handle_data(self, d):
		self.fed.append(d)
	def get_data(self):
		return ''.join(self.fed)

def strip_tags(html):
	s = MLStripper()
	s.feed(html)
	return s.get_data()

def removeNonAsciiChars(str):
	printable = set(string.printable)
	return filter(lambda x: x in printable, str)

def to_ascii(txt):
	txt = txt.encode('ascii',errors='ignore')
	txt = txt.decode('ascii',errors='ignore')
	return txt

# https://docs.python.org/2/library/email.message.html#email.message.Message
def getEmailBodyFromMime(mime):
	body = ''
	mime = to_ascii(mime)
	msg = email.message_from_string(mime) # remove non utf-8 chars
	for part in msg.walk():
		if part.get_content_type() == 'text/plain':
			body = part.get_payload()
		elif part.get_content_type() == 'text/html':
			body = part.get_payload()
			body = strip_tags(body) # remove html tags

	return body

def getAttributesByFrequency(df):

	# bag of words: https://en.wikipedia.org/wiki/Bag-of-words_model

	# MIME
	# https://www.w3.org/Protocols/rfc1341/7_2_Multipart.html

	# MIME parser
	# https://docs.python.org/2/library/email.parser.html

	frequencies = {}
	i = 1
	for index, row in df.iterrows():

		# if i > 1000:
		# 	break
		# else:
		# 	i = i + 1

		msg_class = row['class']

		# msg = getEmailBodyFromMime(row['text']) # remove non utf-8 chars
		msg = row['mime_body']

		def remove_characters(str, chars):
			str = re.sub(r'^http.?:\/\/.*[\r\n]*', '', str, flags=re.MULTILINE)
			for c in chars:
				str = str.replace(c, ' ')
			return str

		msg = remove_characters(msg, {'\\','(',')','-','\'','=','<','>',',','.',':','#','!','www','/','[',']'})
		msg = msg.lower()

		words = msg.split()
		quantity_words = len(set(words))

		for word in set(words):

			if len(word) <= 2 or len(word) >= 20 or word.isdigit(): continue

			if word in frequencies:
				freq = frequencies[word];
				if msg_class == "spam":
					freq[0] = freq[0] + 1
				else:
					freq[1] = freq[1] + 1 
			else:
				if msg_class == "spam":
					frequencies[word] = [1,0]
				else:
					frequencies[word] = [0,1]


	dictOrd = OrderedDict(sorted(frequencies.items(), key=lambda e: abs(e[1][0] - e[1][1]), reverse=True)) # cuidado aca, no lo pondere porque hay el mismo numero de ham spam

	words = []

	i = 1
	for key, value in dictOrd.items():
		if i == 200:
			break
		print(key , value)
		i = i + 1
		words.append(key)

	return words

# rebuild = True if features from data frequency must be rebuilt.
def buildFeaturesFromData(df, rebuild = False):

	df['mime_body'] = list(map(getEmailBodyFromMime, df.text))

	# begin attribute extraction
	# email length
	df['len'] = list(map(lambda mime_body: len(mime_body), df.mime_body))

	# amount of spaces in email
	def count_spaces(str): return str.count(" ")
	df['count_spaces'] = list(map(count_spaces, df.mime_body))

	# amount of closing tags in email
	closing_tags = re.compile('.*</.*|.*/>.*')
	def has_closing_tags(str):
		if closing_tags.match(str.replace('\n',' ')):
			return True
		else:
			return False
	df['has_tags'] = list(map(has_closing_tags, df.mime_body))

	# binary variable that is 1 if the email has any links
	links = re.compile('.*href=.*')

	def has_links(str):
		if links.match(str.replace('\n',' ')):
			return True
		else:
			return False

	df['has_links'] = list(map(has_links, df.mime_body))

	is_capital = re.compile('[A-Z]')

	def count_capitals(str):
		capitals = 0
		for i in str:
			if is_capital.match(i.replace('\n',' ')):
				capitals = capitals + 1
		return capitals

	df['q_capitals'] = list(map(count_capitals, df.mime_body))

	def capitals_in_row(str):
		cap_aux = 0
		in_row = True
		max_in_row = 0
		for i in str:
			if i == ' ':
				continue
			if is_capital.match(i.replace('\n','')) and in_row:
				cap_aux = cap_aux + 1
			elif is_capital.match(i.replace('\n','')) and not in_row:
				cap_aux = 1
				in_row = True
			else:
				max_in_row = max(max_in_row, cap_aux)
				cap_aux = 0
				in_row = False

		max_in_row = max(max_in_row, cap_aux)
		return max_in_row

	df['max_capitals'] = list(map(capitals_in_row, df.mime_body))

	# amount of multiparts
	def mime_multipart(mime):
		mime = to_ascii(mime)
		msg = email.message_from_string(mime)
		i = 0
		for part in msg.walk():
			i = i+1;
		return i

	df['multipart'] = list(map(mime_multipart, df.text))

	if rebuild:
		features = getAttributesByFrequency(df)
	else:
		# these are attributes we gathered using getAttributesByFrequency. we pre-calculate them to avoid having to wait every run.
		features = ['please','original message', 'thanks', 'any', 'attached', 'questions', 'call', 'gas', 'date', 'corp', 'file',
		'energy', 'need', 'meeting', 'group', 'power', 'following', 'there', 'final', 'should', 'more', 'schedule',
		'review', 'think', 'week', 'some', 'deal', 'start', 'scheduling', 'contract', 'money', 'professional', 'been',
		'last', 'work', 'schedules', 'issues', 'viagra', 'however', 'contact', 'thank', 'between', 'solicitation', 'comments',
		'sex', 'messages', 'discuss', 'software', 'save', 'received', 'site', 'changes', 'txt', 'advertisement', 'parsing', 'prices',
		'morning', 'click', 'sure', 'visit', 'stop', 'only', 'working', 'next', 'trading', 'plan', 'tomorrow',
		'awarded', 'soft', 'detected', 'now', 'like', 'about', 'doc', 'who', 'windows', 'basis', 'online', 'product', 'conference',
		'prescription', 'products', 'best', 'fyi', 'point', 'agreement', 'regarding', 'forward', 'north', 'family', 'world', 'team',
		'process', 'help', 'cialis', 'adobe', 'down', 'results', 'thousand', 'first', 'issue', 'link', 'offers', 'note',
		'scheduled', 'management', 'capacity', 'market', 'bill', 'employees', 'daily', 'dollars', 'offere','offered','offer', 'OFFER', 'Nigeria', 'nigeria', 'pills', 'discount']

	# print(features)

	# set extracted features by frequency
	for feature in features:
		df[feature] = list(map(lambda mime_body: feature in mime_body, df.mime_body))

def performance_measure(predictions, actual_classes):

	if len(predictions) != len(actual_classes):
		raise ValueError('Invalid vector size.')

	true_positives  = 0
	false_positives = 0
	false_negatives = 0
	true_negatives  = 0

	for p in range(len(predictions)):
		if predictions[p]  == 'spam' and actual_classes[p] == 'spam':
			true_positives = true_positives + 1
		if predictions[p] == 'spam'  and actual_classes[p] == 'ham':
			false_positives = false_positives + 1
		if predictions[p]  == 'ham'  and actual_classes[p] == 'spam':
			false_negatives = false_negatives + 1
		if predictions[p]  == 'ham'  and actual_classes[p] == 'ham':
			true_negatives = true_negatives + 1	

	precision = true_positives / (true_positives + false_positives)
	recall    = true_positives / (true_positives + false_negatives)

	return {'tp': true_positives, 'fp': false_positives, 'fn': false_negatives, 'tn': true_negatives, 'precision': precision, 'recall': recall}


	pca = PCA(n_components=100)
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	X_test  = pca.transform(X_test)

def begin_competition():

	if not os.path.isfile('dataset_dev/predict.json'):
		raise ValueError('Competition prediction file missing.')

	rf = RandomForestClassifier(n_estimators=100)
	pca = PCA(n_components=100)

	# for the competition we will use a decision tree classifier
	# check if model is already trained
	if os.path.isfile('training/model_comp_rf.pickle') and os.path.isfile('training/model_comp_pca.pickle'):

		# print('Model already trained!')

		with open('training/model_comp_rf.pickle', 'rb') as f:
			rf = pickle.load(f)

		with open('training/model_comp_pca.pickle', 'rb') as f:
			pca = pickle.load(f)

	else:

		# print('Retraining model!') # whole dataset

		ham_txt  = json.load(open('dataset_dev/ham_dev.json'))
		spam_txt = json.load(open('dataset_dev/spam_dev.json'))

		df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
		df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]	

		buildFeaturesFromData(df)

		# prepare data and train classifiers
		select_cols = df.columns.tolist()
		select_cols.remove('class')
		select_cols.remove('text')
		select_cols.remove('mime_body')	

		# print(df)
		X = df[select_cols].values
		y = df['class']

		pca.fit(X)
		X = pca.transform(X)
		rf.fit(X, y)

		with open('training/model_comp_rf.pickle', 'wb') as f:
			pickle.dump(rf, f)

		with open('training/model_comp_pca.pickle', 'wb') as f:
			pickle.dump(pca, f)

	# now that the model is fit, predict a class!
	predict_txt = json.load(open('dataset_dev/predict.json'))
	p_df = pd.DataFrame(predict_txt, columns=['text'])
	buildFeaturesFromData(p_df)

	# prepare data and train classifiers
	select_cols = p_df.columns.tolist()
	# select_cols.remove('class')
	select_cols.remove('text')
	select_cols.remove('mime_body')	

	X_predict = p_df[select_cols].values
	X_predict = pca.transform(X_predict) # build principal components

	predictions = rf.predict(X_predict)

	for p in predictions:
		print(p)

if __name__ == "__main__":

	if BEGIN_COMPETITION:
		begin_competition()
		sys.exit(0)

	# read emails from json
	ham_txt  = json.load(open('dataset_dev/ham_dev.json'))
	spam_txt = json.load(open('dataset_dev/spam_dev.json'))

	ham_len  = len(ham_txt)
	spam_len = len(spam_txt)

	# create pandas dataframe (http://pandas.pydata.org/)
	df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
	df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

	# create test set and validation set. the test set will have 10% of the data.
	# maybe a good idea would be to take 10% of the data in each set (ham and spam)
	# instead of doing general sampling to get a more representative dataset.
	random_state = 0 # set seed to always have the same data split
	df, test = train_test_split(df, test_size = 0.1)

	# build competition dataset to test
	# test.to_json('dataset_dev/predict.json')

	# in the future, we should be able to just save the model parameters
	# instead of having to retrain the whole thing
	with Timer('Build Features From Train Data'):
		buildFeaturesFromData(df)

	with Timer('Build Features From Test Data'):
		buildFeaturesFromData(test)

	# other proposed features:
	# Contains characters other than UTF8?
	# type of file attached
	# exclamation marks
	# importar alguna libreria de NLP, y probar?
	# hacer grafico de keywords (freq en ham vs freq en spam)
	# hacer analisis de palabras en los subjects de los mails (usar libreria de MIME)

	# prepare data and train classifiers
	select_cols = df.columns.tolist()
	# select_cols.remove('class')
	select_cols.remove('text')
	select_cols.remove('mime_body')	

	# print(df)
	X_train = df[select_cols].values
	y_train = df['class']

	X_test  = test[select_cols].values
	y_test  = test['class']

	# dimensionality reduction

	# PCA (no encontré la forma de aplicarlo bien a nuestro caso)
	# pca = PCA(n_components='mle')
	pca = PCA(n_components=100)
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	X_test  = pca.transform(X_test)

	# Feature selection

	# Univariate feature selection (no encontré mejoras incluso con percentil = 99 ni sacando solo un atributo)
	# selector = SelectPercentile(f_classif, percentile=90)
	# selector.fit(X_train, y_train)
	# X_train = selector.transform(X_train)
	# X_test  = selector.transform(X_test)
	
	# select features according to the k highest scores.
	# selector = SelectKBest(f_classif, k=100)
	# selector.fit(X_train, y_train)
	# X_train = selector.transform(X_train)
	# X_test  = selector.transform(X_test)

	# L1-based feature selection (tampoco encontré moejoras por ahora)
	# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
	# model = SelectFromModel(lsvc, prefit=True)
	# X_train = model.transform(X_train)
	# X_test  = model.transform(X_test)

	# Training using 10 fold CV.
	with Timer('Decision Tree Classifier'):
		dtc = DecisionTreeClassifier()
		res = cross_val_score(dtc, X_train, y_train, cv=10)

	print("Cross validation: ", np.mean(res), np.std(res))

	with Timer('Decision Tree Classifier fit'):
		dtc.fit(X_train, y_train)

	# Save model
	with open('training/modelo_dtc.pickle', 'wb') as f:
		pickle.dump(dtc, f)
	
	print("Test set mean accuracy:", dtc.score(X_test,y_test))
	predictions_test = dtc.predict(X_test)
	y_test_list = list(y_test)
	print(performance_measure(predictions_test, y_test_list))

	# Random Forest Classifier
	with Timer('Random Forest Classifier'):
		rf = RandomForestClassifier(n_estimators=100)
		res = cross_val_score(rf, X_train, y_train, cv=10)
	print("Cross validation: ", np.mean(res), np.std(res))
	
	with open('training/modelo_rf.pickle', 'wb') as f:
		pickle.dump(rf, f)
		
	with Timer('Random Forest fit'):
		rf.fit(X_train, y_train)

	print("Test set mean accuracy:", rf.score(X_test,y_test))
	predictions_test = rf.predict(X_test)
	y_test_list = list(y_test)
	print(performance_measure(predictions_test, y_test_list))

	# Naive Bayes
	print("Naive Bayes")
	with Timer('Naive Bayes with Gaussian probabilities'):
		gnb = GaussianNB()
		res = cross_val_score(gnb, X_train, y_train, cv=10)
	print("Cross validation: ", np.mean(res), np.std(res))

	with open('training/modelo_gnb.pickle', 'wb') as f:
		pickle.dump(gnb, f)

	with Timer('Naive Bayes fit'):
		gnb.fit(X_train, y_train)

	print("Test set mean accuracy:", gnb.score(X_test,y_test))
	predictions_test = gnb.predict(X_test)
	y_test_list = list(y_test)
	print(performance_measure(predictions_test, y_test_list))

	# KNN
	print("KNN")
	with Timer('K Nearest Neighbours'): # creo que no tiene sentido
		neigh = KNeighborsClassifier(n_neighbors=20)
		res = cross_val_score(neigh, X_train, y_train, cv=10)
		print("Cross validation: ", np.mean(res), np.std(res))
	with open('training/modelo_knn.pickle', 'wb') as f:
		pickle.dump(neigh, f)

	with Timer('KNN fit'):
		neigh.fit(X_train, y_train)

	print("Test set mean accuracy:", neigh.score(X_test,y_test))
	predictions_test = neigh.predict(X_test)
	y_test_list = list(y_test)
	print(performance_measure(predictions_test, y_test_list))

	# SVM
	# with Timer('Support Vector Machine (SVM)'): # no tiene sentido
	# 	svc = svm.SVC()
	# 	res = cross_val_score(svc, X_train, y_train, cv=10)
	# print(np.mean(res), np.std(res))

	# AdaBoost
	with Timer('AdaBoost Classifier'):
		ada = AdaBoostClassifier(n_estimators=100)
		res = cross_val_score(ada, X_train, y_train, cv=10)
	
	print("Cross validation: ", np.mean(res), np.std(res))
		
	with Timer('AdaBoost fit'):
		ada.fit(X_train, y_train)

	with open('training/modelo_ada.pickle', 'wb') as f:
		pickle.dump(ada, f)

	print("Test set mean accuracy:", ada.score(X_test,y_test))
	predictions_test = ada.predict(X_test)
	y_test_list = list(y_test)
	print(performance_measure(predictions_test, y_test_list))

	# Model selection and evaluation using tools, such as grid_search.GridSearchCV and 
	# cross_validation.cross_val_score, take a scoring parameter that controls what metric they apply to the estimators evaluated.
	# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
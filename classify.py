# Aprendizaje Automatico - DC, FCEN, UBA
# Segundo cuatrimestre 2016

import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

import enchant
import email
import string
import re
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
# import validators

def removeNonAsciiChars(str):
	printable = set(string.printable)
	return filter(lambda x: x in printable, str)

def getAttributesByFrequency(df):

	# bag of words: https://en.wikipedia.org/wiki/Bag-of-words_model
	d = enchant.Dict("en_US")
	# pip install pyenchant

	# MIME
	# https://www.w3.org/Protocols/rfc1341/7_2_Multipart.html

	# https://docs.python.org/2/library/email.parser.html
	frequencies = {}
	i = 1
	for index, row in df.iterrows():

		if i > 1000:
			break
		else:
			i = i + 1

		# print row['c1'], row['c2']

		try:
			msg = email.message_from_string(removeNonAsciiChars(row['text']))
		except:
			print "Skipped message!"
			print row['text']
			continue

		msg_class = row['class']
		# print msg.keys()
		
		# not multipart - i.e. plain text, no attachments, keeping fingers crossed
		body = msg.get_payload(decode=True)

		if msg.is_multipart():
		    for part in msg.walk():
		        ctype = part.get_content_type()
		        cdispo = str(part.get('Content-Disposition'))

		        # skip any text/plain (txt) attachments
		        if ctype == 'text/plain' and 'attachment' not in cdispo:
		            body = part.get_payload(decode=True)  # decode
		            break


		try:
			words = body.split()
		except:
			print msg
			continue

		for word in set(words):

			word = word.strip(',.-')
			word = word.lower()
			# word = filter(lambda x: x in printable, word)

			if len(word) <= 2: continue

			try:
				if not d.check(word): continue
			except:
				continue

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


	dictOrd = OrderedDict(sorted(frequencies.items(), key=lambda e: abs(e[1][0] - e[1][1] ), reverse=True)) # cuidado aca, no lo pondere porque hay el mismo numero de ham spam

	words = []

	for i in range(1,100):
		print i, dictOrd.items()[i]
		word = dictOrd.items()[i][0]
		words.append(word)

	return words


if __name__=="__main__":

	# Leo los mails (poner los paths correctos).
	ham_txt  = json.load(open('dataset_dev/ham_dev.json'))
	spam_txt = json.load(open('dataset_dev/spam_dev.json'))

	ham_len  = len(ham_txt)
	spam_len = len(spam_txt)

	# Imprimo un mail de ham y spam como muestra.
	#print ham_txt[0]
	#print "------------------------------------------------------"
	#print spam_txt[0]
	#print "------------------------------------------------------"

	# Armo un dataset de Pandas 
	# http://pandas.pydata.org/
	df = pd.DataFrame(ham_txt+spam_txt, columns=['text'])
	df['class'] = ['ham' for _ in range(len(ham_txt))]+['spam' for _ in range(len(spam_txt))]

	df = df.sample(frac=1).reset_index(drop=True)

	# Extraigo dos atributos simples: 
	# 1) Longitud del mail.
	df['len'] = map(len, df.text)

	closing_tags = re.compile('.*</.*|.*/>.*')
	links = re.compile('.*href=.*')
	def count_spaces(txt): return txt.count(" ")

	def has_closing_tags(x):
		if closing_tags.match(x.replace('\n',' ')):
			return True
		else:
			return False

	def has_links(x):
		if links.match(x.replace('\n',' ')):
			return True
		else:
			return False

	# 2) Cantidad de espacios en el mail.
	df['count_spaces'] = map(count_spaces, df.text)

	# df['has_tags'] = map(has_closing_tags, df.text)
	# df['has_links'] = map(has_links, df.text)


	# msg = email.message_from_string(spam_txt[0])

	# print msg.keys()

	# print msg.items()

	# for part in msg.walk():
	# 	print part.get_content_type()

	# print msg

	def mime_multipart(mime_txt):
		try:
			msg = email.message_from_string(removeNonAsciiChars(mime_txt))
			return msg.is_multipart()
		except Exception,e:
			print e


			# print mime_txt
			# print "-"*100
			# return 0

	df['multipart'] = map(mime_multipart, df.text)

	# settear features para ahorrar tiempo, para recalcular usar lo de arriba
	features = ['please','the','thanks', 'original', 'message', 'know', 'will', 'this', 'public', 'call', 'have', 'let', 'that', 'attached', 'questions', 
	'any', 'would', 'gas', 'for', 'server', 'exchange', 'converted', 'format', 'meeting', 'group', 'energy', 'power', 'week', 'are', 'your', 'version', 
	'schedule', 'has', 'need', 'there', 'review', 'final', '2002', 'not', 'corp', 'but', 'money', 'deal', 'with', 'should', 'following', 'they', 'think', 
	'also', 'contract', 'john', 'work', 'mark', 'only', '2005', 'file', 'you', 'forwarded', 'issues', 'start', 'however', 'our', '2004', 'professional', 
	'what', 'some', 'last', 'back', 'plan', 'trading', 'more', 'between', 'well', 'still', 'them', 'log', 'stocks', 'statements', 'investing', 'comments', 
	'see', 'software', 'advice', 'morning', 'discuss', 'been', 'adobe', 'when', 'today', 'thank', '2000', 'said', 'agreement', 'was', 'data', 'mike',
	 'received', 'changes', 'had', 'discount', 'deal', 'opportunity', 'chance', 'click', 'product', 'policy', 'order', 'special', 'credit', 'freebies', 
	 'free', 'bucks', 'limited time', 'dream', 'unknown', 'xxx', 'teen', 'teens', 'ass', 'anal', 'penis', 'viagra', 'Niger','dear', 'see you soon', 'tomorrow', 
	 'told', 'said', 'regards', 'meeting', 'let me know', 'lol', 'thanx', 'customer', 'listprice', '$', '</', 'href', '$']

	# features = getAttributesByFrequency(df)

	print features

	for feature in features:
		df[feature] = map(lambda s: feature in s, df.text)

	# other features:
	# Analyize MIME structure... e.g. len(multipart)
	# Contains characters other than UTF8?, ascii?
	# number of recipients
	# type of file attached
	# exclamation marks
	# importar alguna libreria de NLP, y probar?
	# paralelizar el bag of words para mejorar velocidad (ahora tarda bastante)
	# implementar leave out set para testing (seguramente la libreria lo tiene)
	# hacer grafico de keywords (freq en ham vs freq en spam)
	# Preparo data para clasificar
	select_cols = df.columns.tolist()
	select_cols.remove('class')
	select_cols.remove('text')

	# print df
	X = df[select_cols].values
	y = df['class']

	# Ejecuto el clasificador entrenando con un esquema de cross validation
	# de 10 folds.
	print "Decision Tree Classifier"
	dtc = DecisionTreeClassifier()
	res = cross_val_score(dtc, X, y, cv=10)
	print np.mean(res), np.std(res)
	# salida: 0.687566666667 0.0190878702354  (o similar)

	print "Naive Bayes with Gaussian probabilities"
	gnb = GaussianNB()
	res = cross_val_score(gnb, X, y, cv=10)
	print np.mean(res), np.std(res)

	print "K Nearest Neighbours"
	neigh = KNeighborsClassifier(n_neighbors=5)
	res = cross_val_score(neigh, X, y, cv=10)
	print np.mean(res), np.std(res)

	# print "Support Vector Machine (SVM)"
	# svc = svm.SVC()
	# res = cross_val_score(svc, X, y, cv=10)
	# print np.mean(res), np.std(res)

	# print "Random Forest Classifier"
	# rf = RandomForestClassifier(n_estimators=100)
	# res = cross_val_score(rf, X, y, cv=10)
	# print np.mean(res), np.std(res)

	# Model selection and evaluation using tools, such as grid_search.GridSearchCV and 
	# cross_validation.cross_val_score, take a scoring parameter that controls what metric they apply to the estimators evaluated.
	# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

	# cuando hablemos de Naive Bayes, hablemos de Bayesian Poisoning y como los spammers evitan estos filtros normalmente
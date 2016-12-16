# Aprendizaje Automatico

# set this variable to true for competition
BEGIN_COMPETITION = False
BUILD_DATASET = False

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

# def removeNonAsciiChars(str):
# 	printable = set(string.printable)
# 	return filter(lambda x: x in printable, str)

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

def getEmailTitleFromMime(s):
	first = 'subject:'
	last  = '\n'

	s_orig = s
	s = s.lower()

	try:
		start = s.index( first ) + len( first )
		end = s.index( last, start )
		return to_ascii(s_orig[start:end]).strip().replace(".", "").replace("-", "")
	except ValueError:
		return ''

def getAttributesByFrequency(df, text = 'mime_body'):

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
		msg = row[text]

		def remove_characters(str, chars):
			str = re.sub(r'^http.?:\/\/.*[\r\n]*', '', str, flags=re.MULTILINE)
			for c in chars:
				str = str.replace(c, ' ')
			return str

		msg = remove_characters(msg, {'\\','(',')','-','\'','=','<','>',',','.',':','#','!','www','/','[',']'})
		msg = msg.lower()

		words = msg.split()
		quantity_words = len(set(words))

		# for each word in the email
		for word in set(words):

			if len(word) <= 2 or len(word) >= 20 or word.isdigit(): continue

			if word in frequencies:
				# word was already seen, update frequency
				freq = frequencies[word];
				if msg_class == "spam":
					freq[0] = freq[0] + 1
				else:
					freq[1] = freq[1] + 1
			else:
				# new word, add to dictionary
				if msg_class == "spam":
					frequencies[word] = [1,0]
				else:
					frequencies[word] = [0,1]

	dictOrd = OrderedDict(sorted(frequencies.items(), key=lambda e: abs(e[1][0] - e[1][1]), reverse=True)) # cuidado aca, no lo pondere porque hay el mismo numero de ham spam

	print(dictOrd)
	print('-'*1000)

	words = []

	i = 1
	for key, freq in dictOrd.items():
		value = abs(freq[0] - freq[1])
		if value < 500:
			break;
		
		print(key , value)
		i = i + 1
		words.append(key)

	print(words)

	print(len(words))

	return words

def count_capitals(str):
	is_capital = re.compile('[A-Z]')
	capitals = 0
	for i in str:
		if is_capital.match(i.replace('\n',' ')):
			capitals = capitals + 1
	return capitals

# rebuild = True if features from data frequency must be rebuilt.
def buildFeaturesFromData(df, rebuild = False):

	df['title'] = list(map(getEmailTitleFromMime, df.text))

	# for v in df['title']:
	# 	print(v)

	# getAttributesByFrequency(df, 'title')

	df['title_caps'] = list(map(lambda text: count_capitals(text) / (len(text)+1), df.title))

	df['strange_encoding'] = list(map(lambda x: '?iso' in x or '?utf' in x or '?koi8' in x, df.title))

	df['len_title'] = list(map(lambda title: len(title), df.title))

	features = ['pharmacy','cialis','viagra','best','free','1oo%','money','quality','gadgets','$','penis','cheap','sex','stock',
	'hot','meds','best','free','cheap','best','here','draft','revised','schedule','access','deal','buy','phar','cialis','pro']


	for feature in features:
		df[feature+'_title'] = list(map(lambda mime_body: feature in mime_body.lower() or feature in mime_body.lower().replace("0", "o").replace("3", "e"), df.title))


	df['title_length'] = list(map(len, df.title))

	df['mime_body'] = list(map(getEmailBodyFromMime, df.text))

	# begin attribute extraction
	# email length
	df['len'] = list(map(lambda mime_body: len(mime_body), df.mime_body))

	# amount of spaces in email
	df['count_spaces'] = list(map(lambda x: x.count(" "), df.mime_body))

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

	df['q_capitals'] = list(map(count_capitals, df.mime_body))

	is_capital = re.compile('[A-Z]')

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

	df['at'] = list(map(lambda x: x.count('@'), df.text))

	if rebuild:
		features = getAttributesByFrequency(df)
	else:
		# these are attributes we gathered using getAttributesByFrequency. we pre-calculate them to avoid having to wait every run.
		features = ['subject', 'please', 'original message', 'the', 'sent', 'from', 'thanks', 'will', 'have', 'this', 'know', 'that',
		'for', 'let', 'enron', 'would', 'any', 'attached', 'questions', 'call', 'are', 'here', 'with', 'gas', 'date', 'corp', 'has',
		'monday', 'file', 'friday', 'and', 'houston', 'october', 'wednesday', 'not', 'energy', 'you', 'meeting', 'tuesday', 'need',
		'thursday', 'group', 'california', 'power', 'but', 'following', 'there', 'november', 'should', 'final', 'was', 'more', 'they',
		'schedule', 'review', 'think', 'john', 'http', 'week', 'see', 'some', 'forwarded', 'hour', 'when', 'deal', 'ect', 'start',
		'scheduling', 'portland', 'what', 'money', 'been', 'contract', 'fax', 'professional', 'may', 'mark', 'hou', 'last', 'work',
		'contact', 'iso', 'issues', 'schedules', 'log', 'vince', 'viagra', 'still', 'however', 'hourahead', 'can', 'solicitation',
		'back', 'between', 'comments', 'mailto', 'also', 'discuss', 'thank', 'software', 'january', 'messages', 'save', 'were',
		'mike', 'your', 'received', 'site', 'had', 'changes', 'advertisement', 'size', 'could', 'smith', 'prices', 'txt',
		'westdesk', 'parsing', 'morning', 'louise', 'david', 'sure', 'well', 'going', 'click', 'working', 'visit', 'font',
		'only', 'jeff', 'trading', 'stop', 'next', 'like', 'plan', 'did', 'tomorrow', 'now', 'ancillary', 'variances', 'awarded',
		'them', 'windows', 'soft', 'their', 'detected', 'basis', 'steve', 'doc', 'september', 'about', 'who', 'north', 'regarding',
		'product', 'best', 'agreement', 'online', 'forward', 'conference', 'texas', 'fyi', 'prescription', 'products', 'point',
		'family', 'process', 'these', 'team', 'world', 'adobe', 'results', 'kaminski', 'help', 'cialis', 'corel', 'thousand',
		'issue', 'first', 'day', 'down', 'both', 'link', 'offers', 'note', 'market', 'management', 'many', 'employees', 'kevin',
		'michael', 'scheduled', 'his', 'capacity', 'august', 'which', 'daily', 'bill', 'photoshop', 'end', 'data', 'request',
		'dollars', 'probably', 'chris', 'james', 'below', 'two', 'act', 'him', 'microsoft', 'during', 'pipeline', 'future',
		'february', 'body', 'featured', 'bob', 'top', 'said', 'contracts', 'today', 'anyone', 'status', 'yesterday', 'firm',
		'december', 'year', 'special', 'how', 'ena', 'done', 'copies', 'prohibited', 'color', 'creative', 'draft', 'investment',
		'disclosure', 'acrobat', 'requested', 'delete', 'services', 'put', 'another', 'either', 'robert', 'macromedia', 'into',
		'desk', 'able', 'weight', 'global', 'flash', 'related', 'sender', 'asked', 'jim', 'yourself', 'hope', 'markets', 'each',
		'quality', 'distribution', 'kitchen', 'email', 'update', 'might', 'good', 'discussed', 'stocks', 'phone', 'our', 'set',
		'differ', 'security', 'anything', 'offer', 'attachments', 'talk', 'wish', 'afternoon', 'privileged', 'yet', 'whether',
		'create', 'look', 'low', 'easy', 'available', 'advice', 'uncertainties', 'summary', 'already', 'confidential', 'xls', 
		'meaning', '100%', 'shipping', 'report', 'joe', 'copy', 'until', 'richard', 'cause', 'premiere', 'provide', 'west', 'risk',
		'registered', 'studio', 'additional', 'project', 'cell', 'recipient', 'private', 'safe', 'mary', 'where', 'proposed', 'america',
		'trying', 'employee', 'error', 'contain', 'meet', 'life', 'watch', 'credit', 'needs', 'otherwise', 'changed', 'intended', 'popular',
		'possible', 'dave', 'time', 'currently', 'examples', 'does', 'together', 'statements', 'earlier', 'jones', 'follows', 'getting',
		'approval', 'information', 'several', 'continue', 'points', 'likely', 'sex', 'told', 'herein', 'early', 'option', 'million',
		'groups', 'april', 'doing', 'question', 'scott', 'pills', 'legal', 'tabs', 'july', 'oct', 'php', 'concerns', 'com', 'hello',
		'given', 'resources', 'march', 'off', 'those', 'because', 'return', 'case', 'ferc', 'pack', 'investing', 'looks', 'around',
		'financial', 'run', 'williams', 'tom', 'suite', 'generic', 'illustrator', 'street', 'revised', 'hundred', 'again', 'evidence',
		'make', 'research', 'advisor', 'office', 'materially', 'men', 'wanted', 'brand', 'term', 'than', 'without', 'man', 'words',
		'greg', 'volumes', 'securities', 'meds', 'huge', 'addition', 'ever', 'mailings', 'new', 'support', 'something', 'transactions',
		'seek', 'relevant', 'committee', 'inherent', 'york', 'thought', 'weeks', 'africa', 'free', 'discussion', 'sunday', 'deals',
		'price', 'newsletter', 'michelle', 'dose', 'unit', 'listed', 'job', 'operations', 'period', 'stock', 'senior', 'every', 'remove',
		'role', 'send', 'someone', 'failed', 'don', 'fast', 'sorry', 'give', 'posted', 'understand', 'draw', 'close', 'pill', 'transmission',
		'biz', 'counterparty', 'nov', 'appropriate', 'dreamweaver', 'non', 'drugs', 'use', 'event', 'electric', 'involve', 'system', 'kim',
		'none', 'subscribers', 'since', 'plant', 'against', 'perfect', 'predictions', 'readers', 'daren', 'soon', 'objectives', 'paul',
		'within', 'affiliates', 'indicating', 'then', 'flow', 'authorized', 'lisa', 'attend', 'presentation', 'projections', 'keep',
		'21b', 'things', 'davis', 'country', 'sole', 'commission', 'utilities', 'risks', 'fireworks', 'ken', 'decision', 'updated',
		'assumptions', 'room', 'agree', 'beliefs', 'find', 'executive', 'although', 'presently', 'mmbtu', 'appreciate', 'choose',
		'anticipated', 'taylor', 'worldwide', 'else', 'discreet', 'binding', 'relied', 'susan', '$79', 'sally', 'number', 'center',
		'express', 'analyst', 'general', 'after', 'seems', 'outside', 'manager', 'costs', 'couple', 'rick', 'staff', 'beginning',
		'respect', 'called', 'entity', 'boost', 'construed', 'far', 'other', 'least', 'notify', 'hours', 'others', 'leave', 'miller',
		'enforceable', 'london', 'later', 'move', 'saturday', 'estoppel', 'brian', 'options', 'san', 'search', 'mail', 'cheaper',
		'filing', 'under', 'hereto', 'person', 'electricity', 'dealer', 'utility', 'along', 'jeffrey', 'physical', 'added', 'bottom',
		'george', 'says', 'imageready', 'wholesale', 'agreed', 'john', 'david', 'much', '$129', 'importance', 'coming', 'chairman',
		'remain', 'style', 'shares', 'premium', 'guys', 'deciding', 'having', 'named', 'mobile', '$89', 'pertaining', 'guarantee',
		'being', 'individual', 'confirmed', 'volume', 'mid', 'hpl', 'brown', 'spoke', 'regulatory', 'super', 'assistant', 'current',
		'commodity', 'closed', 'determine', 'arial', 'paso', 'gary', 'william', 'estimates', 'position', 'natural', 'href', 'spreadsheet',
		'text', '$179', 'strictly', 'advise', 'problem', 'advises', 'trader', 'east', 'tim', 'open', 'statements', 'positions', 'part',
		'forward', 'symbol', 'sheet', 'yours', 'sum', 'pain', 'vice', 'everyone', 'profile', 'appears', 'asset', 'expectations',
		'tadalafil', 'problems', 'pg&e', 'due', 'numbers', 'month', 'load', 'unable', 'trades', 'way', 'harris', 'interview', 'cannot',
		'different', 'list', 'accuracy', 'over', 'performance', 'analysis', 'decoration', 'traders', 'facts', 'contains', 'floor',
		'regards', 'chief', 'eol', 'medical', 'websites', 'preferred', 'full', 'past', 'messaging', 'transportation', 'foresee',
		'order', 'reports', 'looking', 'hot', 'might', 'feedback', 'reporting', 'expects', 'signed', 'members', 'lot', 'dynegy',
		'represent', 'identified', 'response', 'believe', 'reform', 'dear', 'speculative', 'really', 'rather', 'responsible',
		'associated', 'shirley', 'meter', 'sites', 'pro', 'internet', 'box', 'mix', 'sans', 'store', 'professional', 'professional',
		'account', 'add', 'uns', 'reported', 'filed', 'version', 'heard', 'included', 'reached', 'may', 'respond', 'service',
		'settlement', 'required', 'critical', 'growth', 'karen', 'comment', 'university', 'storage', 'agreements', 'states', 'she'
		'level', 'proven', 'wants', 'health', 'info', 'section', 'eric', 'managing', 'steven', 'model', 'washington', 'talked', 'allen',
		'occur', 'course', 'understood', 'transwestern', 'china', 'bid', 'cut', 'compliance', 'intervention', 'shop', 'conflict',
		'privacy', 'historical', 'previous', 'spam', 'unless', 'ensure', 'graphics', 'sept', 'activities', 'lose', 'short', 'martin',
		'erections', 'weather', 'believes', 'ets', 'sincerely', 'paid', 'beck', 'involved', 'xanax', 'ahead', 'frank', 'pass', 'once',
		'dan', 'central', 'removed', 'thoughts', 'actual', 'titles', 'sir', 'untitled', 'learn', 'suggested', 'helvetica', 'night',
		'continuing', 'erection', 'structure', 'affiliated', 'very', 'meetings', 'hall', 'weekend', 'expect', 'value', 'allow', 'agenda',
		'lots', 'addressed', 'projects', 'taking', 'increase', 'specials', 'attachment', 'lavorato', 'investors', 'its', 'check',
		'president', 'held', 'met', 'website', 'thomas', 'investor', 'lead', 'pricing', 'haven', 'height', '0rder', 'american',
		'building', 'difficult', 'retail', 'must', 'publisher', 'correct', 'entertainment', 'right', 'immediately', 'just', 'capital',
		'maybe', 'despite', 'lay', 'lunch', 'prepared', 'parties', 'appear', 'commercial', 'internal', 'effects', 'sexual', 'progress',
		'season', 'will', 'demand', 'kind', 'never', 'yes', 'estimates', 'lee', 'assume', 'better', 'international', 'otc', 'sales',
		'post', 'clear', 'second', 'director', 'design', 'farmer', 'ubs', 'factual', 'broker', 'weekly', 'reliable', 'united', 'americas',
		'court', 'thousands', 'late', 'secure', 'master', 'counterparties', 'getresponse', 'waterfall', 'doesn', 'johnson', 'units',
		'balance', 'dec', 'location', 'understands', 'wide', 'representative', 'conversation', 'exchange', 'computer', 'vanessa',
		'planning', 'idea', '70%', 'actually', 'though', 'goals', 'countries', 'anticipates', '80%', 'morgan', 'understanding',
		'summer', 'including', 'pacific', 'administration', 'memo', 'events', 'sell', 'situation', 'cheap', 'selected', 'change',
		'approximately', 'stephen', 'margin', 'approach', 'receive', 'analysts', 'jan', 'asap', 'access', 'limited', 'book',
		'remaining', 'penny', 'dates', 'fact', 'concerned', 'highly', 'claim', 'pharmacy', 'take', 'specific', 'northern',
		'updates', 'hearing', 'estimate', 'valium', 'moving', 'earnings', 'music', 'running', 'copyright', 'amount', 'watson',
		'litigation', 'promotional', 'org', 'moved', 'epmi', 'medications', 'competitors', 'opt', 'pretty', 'isn', 'range',
		'pink', 'width', 'begin', 'penis', 'derivatives', 'completeness', 'simply', 'could', 'impact', 'moore', 'constitutes',
		'millions', 'written', 'actions', 'dvd', 'too', 'session', 'compare', 'compensated', 'rate', 'pat', 'omit', 'perhaps',
		'counsel', 'unique', 'place', 'went', 'press', 'state', 'resume', 'invest', 'unsubscribe', 'indicated', 'fuel', 'steffes',
		'finance', 'previously', 'plants', 'solutions', 'standard', 'licensed', 'consider', 'tell', 'created', 'noted', 'sending',
		'development', 'separate', 'larry', 'spot', 'law', 'directly']

	# print(features)

	# set extracted features by frequency
	for feature in features:
		df[feature] = list(map(lambda mime_body: feature in mime_body.lower(), df.mime_body))

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
		select_cols.remove('title')

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
	select_cols.remove('title')

	X_predict = p_df[select_cols].values
	X_predict = pca.transform(X_predict) # build principal components

	predictions = rf.predict(X_predict)

	for p in predictions:
		print(p)

if __name__ == "__main__":

	if BEGIN_COMPETITION:
		begin_competition()
		sys.exit(0)

	if BUILD_DATASET:
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
		df, test_models = train_test_split(df, test_size = 0.1)

		# build competition dataset to test
		# test.to_json('dataset_dev/predict.json')

		# print(list(map(getEmailTitleFromMime, df.text)))

		# sys.exit(0)

		# in the future, we should be able to just save the model parameters
		# instead of having to retrain the whole thing
		with Timer('Build Features From Train Data'):
			buildFeaturesFromData(df, False)

		with Timer('Build Features From Test Data'):
			buildFeaturesFromData(test, False)
			buildFeaturesFromData(test_models, False)

		# other proposed features:
		# Contains characters other than UTF8?
		# type of file attached
		# exclamation marks
		# importar alguna libreria de NLP, y probar?
		# hacer grafico de keywords (freq en ham vs freq en spam)
		# hacer analisis de palabras en los subjects de los mails (usar libreria de MIME)

		# prepare data and train classifiers
		select_cols = df.columns.tolist()
		select_cols.remove('class')
		select_cols.remove('text')
		select_cols.remove('mime_body')	
		select_cols.remove('title')

		X_train = df[select_cols].values
		y_train = df['class']

		X_test  = test[select_cols].values
		y_test  = test['class']

		X_test_models = test[select_cols].values
		y_test_models = test['class']

		np.save('X_train', X_train)
		np.save('y_train', y_train)
		np.save('X_test', X_test)
		np.save('y_test', y_test)
		np.save('X_test_models', X_test_models)
		np.save('y_test_models', y_test_models)

	else:
		X_train = np.load('X_train.npy')
		y_train = np.load('y_train.npy')

		X_test  = np.load('X_test.npy')
		y_test  = np.load('y_test.npy')

		X_test_models = np.load('X_test_models.npy')
		y_test_models = np.load('y_test_models.npy')

	# dimensionality reduction

	# PCA (no encontré la forma de aplicarlo bien a nuestro caso)
	# pca = PCA(n_components='mle')
	
	# pca = PCA(n_components=100)
	# pca.fit(X_train)
	# X_train = pca.transform(X_train)
	# X_test  = pca.transform(X_test)

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

	# Naive Bayes
	# print("Naive Bayes")
	# with Timer('Naive Bayes with Gaussian probabilities'):
	# 	gnb = GaussianNB()
	# 	res = cross_val_score(gnb, X_train, y_train, cv=10)
	# 	print("Cross validation: ", np.mean(res), np.std(res))

	# with open('training/modelo_gnb.pickle', 'wb') as f:
	# 	pickle.dump(gnb, f)

	# with Timer('Naive Bayes fit'):
	# 	gnb.fit(X_train[:,0:200], y_train)
	# 	print("Test set mean accuracy:", gnb.score(X_test_models[:,0:200],y_test_models))
	# 	predictions_test = gnb.predict(X_test_models[:,0:200])
	# 	y_test_list = list(y_test_models)
	# 	print(performance_measure(predictions_test, y_test_list))

	# with Timer('Naive Bayes fit'):
	# 	gnb.fit(X_train[:,0:500], y_train)
	# 	print("Test set mean accuracy:", gnb.score(X_test_models[:,0:500],y_test_models))
	# 	predictions_test = gnb.predict(X_test_models[:,0:500])
	# 	y_test_list = list(y_test_models)
	# 	print(performance_measure(predictions_test, y_test_list))

	# with Timer('Naive Bayes fit'):
	# 	gnb.fit(X_train, y_train)
	# 	print("Test set mean accuracy:", gnb.score(X_test_models,y_test_models))
	# 	predictions_test = gnb.predict(X_test_models)
	# 	y_test_list = list(y_test_models)
	# 	print(performance_measure(predictions_test, y_test_list))

	for trees in [100]: #[30, 50, 100, 150, 200]:

		print('Trees: {}'.format(trees))

		# Random Forest Classifier
		with Timer('Random Forest Classifier'):
			rf = RandomForestClassifier(n_estimators=trees)
			# res = cross_val_score(rf, X_train, y_train, cv=10)
		# print("Cross validation: ", np.mean(res), np.std(res))
		
		# with open('training/modelo_rf.pickle', 'wb') as f:
		# 	pickle.dump(rf, f)
			
		with Timer('Random Forest fit'):
			rf.fit(X_train, y_train)

		print("Test set mean accuracy:", rf.score(X_test, y_test))
		predictions_test = rf.predict(X_test)
		y_test_list = list(y_test)
		print(performance_measure(predictions_test, y_test_list))

	# # AdaBoost
	# with Timer('AdaBoost Classifier'):
	# 	ada = AdaBoostClassifier(n_estimators=150)
	# 	# res = cross_val_score(ada, X_train, y_train, cv=10)
	
	# # print("Cross validation: ", np.mean(res), np.std(res))
		
	# with Timer('AdaBoost fit'):
	# 	ada.fit(X_train, y_train)

	# with open('training/modelo_ada.pickle', 'wb') as f:
	# 	pickle.dump(ada, f)

	# print("Test set mean accuracy:", ada.score(X_test,y_test))
	# predictions_test = ada.predict(X_test)
	# y_test_list = list(y_test)
	# print(performance_measure(predictions_test, y_test_list))

	# # Training using 10 fold CV.
	# with Timer('Decision Tree Classifier'):
	# 	dtc = DecisionTreeClassifier()
	# 	# res = cross_val_score(dtc, X_train, y_train, cv=10)

	# # print("Cross validation: ", np.mean(res), np.std(res))

	# with Timer('Decision Tree Classifier fit'):
	# 	dtc.fit(X_train, y_train)

	# # Save model
	# with open('training/modelo_dtc.pickle', 'wb') as f:
	# 	pickle.dump(dtc, f)
	
	# print("Test set mean accuracy:", dtc.score(X_test,y_test))
	# predictions_test = dtc.predict(X_test)
	# y_test_list = list(y_test)
	# print(performance_measure(predictions_test, y_test_list))

	# KNN
	print("KNN")
	for k in [5, 10, 20]:
		print('K: {}'.format(k))
		with Timer('K Nearest Neighbours'): # creo que no tiene sentido
			neigh = KNeighborsClassifier(n_neighbors=k)
			# res = cross_val_score(neigh, X_train, y_train, cv=10)
			# print("Cross validation: ", np.mean(res), np.std(res))
		# with open('training/modelo_knn.pickle', 'wb') as f:
		# 	pickle.dump(neigh, f)

		with Timer('KNN fit'):
			neigh.fit(X_train, y_train)

		print("Test set mean accuracy:", neigh.score(X_test_models,y_test_models))
		predictions_test = neigh.predict(X_test_models)
		y_test_list = list(y_test)
		print(performance_measure(predictions_test, y_test_list))

	# SVM
	# with Timer('Support Vector Machine (SVM)'): # no tiene sentido
	# 	svc = svm.SVC()
	# 	res = cross_val_score(svc, X_train, y_train, cv=10)
	# print(np.mean(res), np.std(res))

	# Model selection and evaluation using tools, such as grid_search.GridSearchCV and 
	# cross_validation.cross_val_score, take a scoring parameter that controls what metric they apply to the estimators evaluated.
	# http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
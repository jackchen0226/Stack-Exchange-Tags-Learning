import csv
import pandas as pd
#import numpy as np
import nltk.text
from nltk import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import defaultdict

Test = pd.read_csv('data/robotics.csv')

def content():
	q_tokens = []
	for i in range(Test.shape[0]):
		# Tokenize words from the title and the content then combine them together.
		t_tokens = word_tokenize(BeautifulSoup(Test['title'].iloc[i], "html5lib").get_text())
		c_tokens = word_tokenize(BeautifulSoup(Test['content'].iloc[i], "html5lib").get_text())
		t_tokens.extend(c_tokens)
		# The index of q_tokens should be equal to the id of the post
		q_tokens.append(t_tokens)
	return q_tokens

def tagCount():
	tagdict = defaultdict(int)
	l = []
	for i in range(len(Test['tags'])):
		l.extend(word_tokenize(Test['tags'].iloc[i]))
	for i in l:
		tagdict[i] += 1
	return tagdict

def wordCount():
	# a list with a list of a posts' word with stopwords filtered out
	stop = set(stopwords.words('english'))
	stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
	rawwords = content()
	words = []
	for i in range(len(rawwords)):
		words.append(rawwords[i])
		for j in rawwords[i]:
			if j in stop:
				words[i].remove(j)

	pass

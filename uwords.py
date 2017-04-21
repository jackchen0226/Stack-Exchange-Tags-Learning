import pandas as pd
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
Train = pd.read_csv("QuoraData/train.csv")
Train.fillna('')
utokens = []
d_len = len(Train)

def uwords():
	l = Train['question2'].tolist()
	tokens = []
	for i in l:
		if type(i) != str:
			i = " "
		tokens = text_to_word_sequence(i)
		for i in tokens:
			if i not in utokens:
				utokens.append(i)

def allsen(data):
	q1 = data['question1'].tolist()
	q2 = data['question2'].tolist()
	out = []

	for i in range(d_len):
		if type(q1[i]) != str:
			q1[i] = ''
		if type(q2[i]) != str:
			q2[i] = ''
		q1[i] = set(text_to_word_sequence(q1[i]))
		q2[i] = set(text_to_word_sequence(q2[i]))

		out.append([q1[i]])
		out[i].append(q2[i])

	return out

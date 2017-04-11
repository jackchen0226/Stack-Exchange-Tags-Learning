import pandas as pd
import numpy as np
from nltk import word_tokenize
Train = pd.read_csv("QuoraData/train.csv")
utokens = []

def uwords():
	l = Train['question1'].tolist()
	tokens = []
	for i in l:
		tokens = word_tokenize(i)
		for i in tokens:
			if i not in utokens:
				utokens.append(i)




import pandas as pd 
import numpy as np 
from nltk import word_tokenize
import pickle

Train = pd.read_csv("QuoraData/train.csv")
def q_dict_gen(dataset, name : str):
	q_dict = {}
	for sen in dataset:
		if type(sen) != str:
			sen = " "
		q_dict[len(q_dict)] = word_tokenize(sen)

	qpkl = open(name+'.pkl', 'wb')
	pickle.dump(q_dict, qpkl, protocol=pickle.HIGHEST_PROTOCOL)
	qpkl.close()
import pandas as pd 
import numpy as np 
from nltk import word_tokenize
import pickle
import click

@click.command()
@click.option('--dataset', help='The name of row of the data')
@click.option('--name', prompt='Choose a name for the file', help='The name of the pickled file')
def q_dict_gen(dataset, name : str):
	q_dict = {}
	Train = pd.read_csv("QuoraData/train.csv")
	for sen in Train[dataset]:
		if type(sen) != str:
			sen = " "
		q_dict[len(q_dict)] = word_tokenize(sen)

	qpkl = open(name+'.pkl', 'wb')
	pickle.dump(q_dict, qpkl, protocol=pickle.HIGHEST_PROTOCOL)
	qpkl.close()

if __name__ == '__main__':
	q_dict_gen()
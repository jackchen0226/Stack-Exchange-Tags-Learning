import keras
import keras.backend as K
import pandas as pd
import numpy as np
from nltk import word_tokenize
import pickle
#import bprofile
from collections import defaultdict


Train = pd.read_csv("QuoraData/train.csv")
Train.fillna('')
#Train = Train[:9600]

def make_train_generator(train_data):
	"""Returns a generator and its period."""
	batch_size = 32
	train_len = len(train_data)
	period = int(np.ceil(train_len / batch_size))

	def generator():
		#profiler = bprofile.BProfile('genpin.png')
		while True:
			i = 0
			while i < train_len:
				#with profiler:
				n = min(batch_size, train_len - i)
				data = np.zeros((n, num_features))
				target = np.zeros((n, 2))
				for qidx in range(n):
					train_row = train_data.iloc[i + qidx]
					data[qidx, :] = get_features(train_row)
					target[qidx, 0] = train_row['is_duplicate'] == 0
					target[qidx, 1] = train_row['is_duplicate'] == 1
				yield data, target
				i += n
	return generator(), period


def generate_features():
	word_to_index = {}
	'''
	pronouns = ['he', 'she', 'it', 'they', 'i']
	qpkl = open('q1.pkl', 'rb')
	q1_tokens = pickle.load(qpkl)
	qpkl.close()

	qpkl = open('q2.pkl', 'rb')
	q2_tokens = pickle.load(qpkl)
	qpkl.close()'''
	
	sent = {}
	for i in range(len(Train)):
		if Train['question2'].iloc[i] != str:
			sent[i] = (set(word_tokenize(Train['question1'].iloc[i])),
						set(''))
		else:
			sent[i] = (set(word_tokenize(Train['question1'].iloc[i])),
						set(word_tokenize(Train['question2'].iloc[i])))

	# Finding unique words
	onecount = defaultdict(int)
	for i in sent.keys():
		for j in sent[i][0]:
			if j not in sent[i][1]:
				onecount[j] += 1
				
		for j in sent[i][1]:
			if j not in sent[i][0]:
				onecount[j] += 1
	bothcount = defaultdict(int)
	for i in sent.keys():
		for j in sent[i][0]:
			if j in sent[i][1]:
				bothcount[j] += 1

	# Trimming words that appear less than 10 times
	okeys = set(onecount.keys())
	for i in okeys:
		if onecount[i] < 10:
			onecount.pop(i)

	bkeys = set(bothcount.keys())
	for i in bkeys:
		if bothcount[i] < 10:
			bothcount.pop(i)


	num_features = 2

	def get_features(row):
		features = np.zeros(num_features)
		# Rewrite, just need features for occurance in word, not this complex!!!
		# 0 : A unique word is in one question but not the other
		# 1 : A unique word is in both questions
		l1 = set(word_tokenize(row['question1']))
		l2 = str(row['question2'])
		for i in l1:
			if i in okeys and i not in l2:
				features[0] = True
			if i in bkeys and i in l2:
				features[1] = True


		return features
	return num_features, get_features


num_features, get_features = generate_features()
def getModel():
	ins = keras.layers.Input((num_features,))
	x = ins
	x = keras.layers.Dense(100)(x)
	x = keras.layers.Activation('relu')(x)
	x = keras.layers.Dense(2)(x)
	x = keras.layers.Activation('softmax')(x)

	model = keras.models.Model(ins, x)
	model.compile(loss='categorical_crossentropy', 
			optimizer=keras.optimizers.Adadelta(),
			metrics=['accuracy'])
	return model


def main():
	k = getModel()
	split = len(Train) * 9 // 10
	gen, gen_len = make_train_generator(Train[:split])
	val_gen, val_len = make_train_generator(Train[split:])
	k.fit_generator(generator=gen, steps_per_epoch=gen_len, epochs=10,
			validation_data=val_gen, validation_steps=val_len)
	#k.evaluate()


if __name__ == '__main__':
	#with bprofile.BProfile("profile2.png"):
	main()

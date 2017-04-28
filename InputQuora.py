import keras
import keras.backend as K
import pandas as pd
import numpy as np
from nltk import word_tokenize
import pickle
import bprofile
from 


Train = pd.read_csv("QuoraData/train.csv")
Train.fillna('')
#Train = Train[:9600]

def make_train_generator(train_data):
	"""Returns a generator and its period."""
	batch_size = 32
	train_len = len(train_data)
	period = int(np.ceil(train_len / batch_size))

	def generator():
		# profiler = bprofile.BProfile('genpin.png')
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


	indup = defaultdict(int)
	idup = defaultdict(int)
	keys = set(onecount.keys())
	for i in range(len(sent)):
		for j in sent[i][0]:
			if j in keys:
				if j not in sent[i][1] and dupes.iloc[i] == 0:
					indup[j] += 1
				elif j not in sent[i][1] and dupes.iloc[i] == 1:
					idup[j] += 1

		for j in sent[i][1]:
			if j in keys:
				if j not in sent[i][0] and dupes.iloc[i] == 0:
					indup[j] += 1
				elif j not in sent[i][0] and dupes.iloc[i] == 1:
					idup[j] += 1
	
	keys = list(onecount.keys())
	for i in keys:
		if onecount[i] < 10:
			onecount.pop(i)

	iflu_words = []
	for i in idup.keys():
		if idup[i] / onecount[i] >= 0.95:
			iflu_words.append(i)
	iflu_words = set(iflu_words)

	influ_words = []
	for i in indup.keys():
		if indup[i] / onecount[i] >= 0.95:
			influ_words.append(i)
	influ_words = set(influ_words)

	num_features = 2

	def get_features(row):
		features = np.zeros(num_features)
		# Rewrite, just need features for occurance in word, not this complex!!!
		for i in sent[row['id']][0]:
			if i not in sent[row['id']][1]:
				if i in iflu_words:
					features[0] = True
				if i in influ_words:
					features[1] = True

		for i in sent[row['id']][1]:
			if i not in sent[row['id']][0]:
				if i in iflu_words:
					features[0] = True
				if i in influ_words:
					features[1] = True

		# Need matching words, pronoun existence as features.
		# 0: no pronouns, 1: pronoun in q1, 2: pronoun in q2, 3: both have pronouns
		"""for i in pronouns:
			if i not in q1_tokens[row['id']] and i not in q2_tokens[row['id']]:
				features[0] = True
			if i in q1_tokens[row['id']] and i not in q2_tokens[row['id']]:
				features[1] = True
			if i not in q2_tokens[row['id']] and i in q2_tokens[row['id']]:
				features[2] = True
			if i in q1_tokens[row['id']] and i in q2_tokens[row['id']]:
				features[3] = True"""
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
	k.fit_generator(generator=gen, steps_per_epoch=gen_len, epochs=1,
			validation_data=val_gen, validation_steps=val_len)
	#k.evaluate()


if __name__ == '__main__':
	#with bprofile.BProfile("profile2.png"):
	main()

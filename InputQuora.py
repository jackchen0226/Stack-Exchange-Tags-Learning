import keras
import keras.backend as K
import pandas as pd
import numpy as np
from nltk import word_tokenize
import pickle
import bprofile

Train = pd.read_csv("QuoraData/train.csv")
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
	pronouns = ['he', 'she', 'it', 'they', 'i']
	qpkl = open('q1.pkl', 'rb')
	q1_tokens = pickle.load(qpkl)
	qpkl.close()

	qpkl = open('q1.pkl', 'rb')
	q2_tokens = pickle.load(qpkl)
	qpkl.close()

	num_features = 4

	def get_features(row):
		features = np.zeros(num_features)
		
		# Need matching words, pronoun existence as features.
		# 0: no pronouns, 1: pronoun in q1, 2: pronoun in q2, 3: both have pronouns
		for i in pronouns:
			if i not in q1_tokens[row['id']] and i not in q2_tokens[row['id']]:
				features[0] = True
			if i in q1_tokens[row['id']] and i not in q2_tokens[row['id']]:
				features[1] = True
			if i not in q2_tokens[row['id']] and i in q2_tokens[row['id']]:
				features[2] = True
			if i in q1_tokens[row['id']] and i in q2_tokens[row['id']]:
				features[3] = True
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
	k.fit_generator(generator=gen, steps_per_epoch=gen_len, epochs=5,
			validation_data=val_gen, validation_steps=val_len)


if __name__ == '__main__':
	#with bprofile.BProfile("profile2.png"):
	main()

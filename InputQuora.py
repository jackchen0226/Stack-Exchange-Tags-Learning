import keras
import keras.backend as K
import pandas as pd
import numpy as np
from nltk import word_tokenize

Train = pd.read_csv("QuoraData/train.csv")
def make_train_generator(train_data):
	"""Returns a generator and its period."""
	batch_size = 32
	train_len = len(train_data)
	period = int(np.ceil(train_len / batch_size))

	def generator():
		while True:
			i = 0
			while i < train_len:
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
	"""Should return the number of features.
	"""
	'''
	q1 = Train["question1"].tolist()
	q2 = Train['question2'].tolist()
	for i in range(len(q1)):
		try:
			q1[i] = word_tokenize(q1[i])
			q2[i] = word_tokenize(q2[i])
		except TypeError:
	'''
	word_to_index = {}
	pronouns = ['he', 'she', 'it', 'they', 'i']

	def get_features(row):
		features = np.zeros(num_features)
		'''
		Need matching words, pronoun existence as features.
		for i in pronouns:
			for j in range(len(Train)):
				if 
			break
		'''
		return features
	return 2, get_features



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


Train = Train[:10000]
def main():
	k = getModel()
	split = len(Train) * 9 // 10
	gen, gen_len = make_train_generator(Train[:split])
	val_gen, val_len = make_train_generator(Train[split:])
	k.fit_generator(generator=gen, steps_per_epoch=gen_len, epochs=10,
			validation_data=val_gen, validation_steps=val_len)


if __name__ == '__main__':
	main()

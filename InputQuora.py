import keras
import keras.backend as K
from keras.preprocessing.text import text_to_word_sequence
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pickle
#import bprofile
from collections import defaultdict
import re
from string import punctuation


Train = pd.read_csv("QuoraData/train.csv")
Train = Train.fillna('')
stop_words = stopwords.words('english')
#Train = Train[:9600]


# Currie32's text cleaning
def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    # text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w.lower() in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text_to_word_sequence(text))


def process_questions(question_dict, questions, question_list_name, dataframe):
    '''transform questions and display progress'''
    try: # See if key in dict exists
            for i, question in enumerate(questions):
            	if type(question_dict[i]) == list:
	                question_dict[i].append(text_to_wordlist(question))
	                if len(question_dict) % 100000 == 0:
	                    progress = len(question_dict)/len(dataframe) * 100
	                    print("{} is {}% complete.".format(question_list_name, round(progress, 1)))
	                    
    except KeyError: # Means dict is empty 
        for i, question in enumerate(questions):
            question_dict[i] = [text_to_wordlist(question)]
            if len(question_dict) % 100000 == 0:
                progress = len(question_dict)/len(dataframe) * 100
                print("{} is {}% complete.".format(question_list_name, round(progress, 1)))


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
					# If doing Test, this needs to be tweaked
					target[qidx, 0] = train_row['is_duplicate'] == 0
					target[qidx, 1] = train_row['is_duplicate'] == 1
				yield data, target
				i += n
	return generator(), period


def generate_features():
	sent = {}
	'''
	for i in range(len(Train)):
		sent[i] = (set(word_tokenize(Train['question1'].iloc[i])),
					set(word_tokenize(Train['question2'].iloc[i])))
	'''
	process_questions(sent, Train.question1, 'Sentences part 1', Train)
	process_questions(sent, Train.question2, 'Sentences part 2', Train)

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

	print(len(onecount))
	print(len(bothcount))
	# len(onecount) should be 18973
	# len(bothcount) should be 8389

	okeys = list(onecount.keys())
	bkeys = list(bothcount.keys())
	num_features = len(onecount) + len(bothcount)
	print(num_features)

	def get_features(row):
		features = np.zeros(num_features) - 1
		# 0-18972 : A unique word is in one question but not the other
		# 18973-27362 : A unique word that appears in both sentences
		l1 = set(text_to_wordlist(row['question1']))
		l2 = set(text_to_wordlist(row['question2']))
		for word in l1:
			if word in okeys and word not in l2:
				features[okeys.index(word)] = True
			if word in bkeys and word in l2:
				features[bkeys.index(word) + len(onecount)] = True
		for word in l2:
			if word in okeys and word not in l1:
				features[okeys.index(word)] = True
		return features
	return num_features, get_features


num_features, get_features = generate_features()
def getModel():
	ins = keras.layers.Input((num_features,))
	x = ins
	x = keras.layers.Dense(50)(x)
	x = keras.layers.Dropout(0.1)(x)
	x = keras.layers.Activation('relu')(x)
	#x = keras.layers.Dropout(0.1)(x)
	x = keras.layers.Dense(2)(x)
	x = keras.layers.Activation('softmax')(x)
	#x = keras.layers.Dropout(0.1)(x)

	model = keras.models.Model(ins, x)
	model.summary()
	model.compile(loss='categorical_crossentropy', 
			optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.5),
			metrics=['accuracy'])
	return model


def main():
	k = getModel()
	split = len(Train) * 9 // 10
	#score = k.predict(x=Train.as_matrix())
	#print(score)
	gen, gen_len = make_train_generator(Train[:split])
	val_gen, val_len = make_train_generator(Train[split:])
	k.fit_generator(generator=gen, steps_per_epoch=gen_len, epochs=3,
			validation_data=val_gen, validation_steps=val_len)

	score = k.predict_generator(generator=gen, steps=gen_len)
	print(score)
	'''
	diff_qids = []
	for i in range(len(score)):
		if score[i][0] < 0.55 and score[i][0] > 0.45:
			diff_qids.append(i)
		elif score[i][1] < 0.55 and score[i][1] > 0.45:
			diff_qids.append(i)
	'''
	qidspkl = open('pickled/score.pkl', 'wb')
	pickle.dump(score, qidspkl, protocol=pickle.HIGHEST_PROTOCOL)
	qidspkl.close()


if __name__ == '__main__':
	#with bprofile.BProfile("profile2.png"):
	main()

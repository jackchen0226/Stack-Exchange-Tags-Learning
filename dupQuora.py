'''
import keras
import keras.backend as K

num_words = K.placeholder()
ins = keras.layers.InputLayer((num_words,))
x = keras.layers.Dense(100)(ins)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Dense(2)(x)
x = keras.layers.Softmax()(x)

model = keras.model.Model(ins, x
model.compile('categorical_crossentropy')
'''
from scipy.io import wavfile
from scipy.fftpack import rfft
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import normalize

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Flatten, Lambda, Dropout, Activation, LSTM, TimeDistributed, Convolution1D, MaxPooling1D

'''
SEED = 42
N_LAYERS = 3
FILTER_LENGTH = 3
CONV_FILTER_COUNT = 16
LSTM_COUNT = 256
BATCH_SIZE = 10
EPOCH_COUNT = 100
'''

def get_ffts(fname):
	samplerate, data = wavfile.read(fname)
	scale = 10 # this value affects window_length
	window_length = int(samplerate/scale)
	N = math.floor(len(data)/window_length)
	data = data[:N * window_length]
	spectrum = np.zeros((N,int(window_length/2)))
	for i in range(0,N):
		curr_data = data[i * window_length : (i+1) * window_length]
		fftOut_curr = np.absolute(rfft(curr_data))
		mags_curr = fftOut_curr[0 : int(window_length/2)]
		spectrum[i] = normalize(mags_curr[:,np.newaxis], axis=0).ravel()
	return spectrum[:,0:samplerate//2]

def train_model(data):

	x = np.array([np.array(data[: -10, :])])
	y = np.array([np.array(data[10: , :])])
	x = np.reshape(x, (x.shape[1],1,x.shape[2]))
	y = np.reshape(y, (y.shape[1],1,y.shape[2]))

	n_features = x.shape[2]
	input_shape = (None, n_features)
	
	'''
	model_input = Input(input_shape, name='input')
	layer = model_input
	for i in range(N_LAYERS):
		# convolutional layer names are used by extract_filters.py
		layer = Convolution1D(
			nb_filter=CONV_FILTER_COUNT,
			filter_length=FILTER_LENGTH,
			name='convolution_' + str(i + 1)
			)(layer)
		layer = Activation('relu')(layer)
		layer = MaxPooling1D(2)(layer)

	layer = Dropout(0.5)(layer)
	layer = LSTM(LSTM_COUNT, return_sequences=True)(layer)
	layer = Dropout(0.5)(layer)
	layer = TimeDistributed(Dense(y.shape[1]))(layer)
	layer = Activation('softmax', name='output_realtime')(layer)
	time_distributed_merge_layer = Lambda(
		function=lambda y: K.mean(y, axis=1), 
		output_shape=lambda shape: (shape[0],) + shape[2:],
		name='output_merged'
		)
	model_output = time_distributed_merge_layer(layer)
	model = Model(model_input, model_output)
	opt = RMSprop(lr=0.00001)
	model.compile(
		loss='cosine_proximity',
		optimizer=opt,
		metrics=['accuracy']
		)
	'''
	
	model = Sequential()
	model.add(Convolution1D(64, 5, padding='same',input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Convolution1D(64, 5))
	model.add(Activation('relu'))
	model.add(MaxPooling1D(4))
	model.add(Dropout(0.25))

	model.add(Convolution1D(64, 3, padding='same'))
	model.add(Activation('relu'))
	model.add(Convolution1D(64, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling1D(2))
	model.add(Dropout(0.25))
	model.add(Dense(400))
	model.add(Activation('softmax'))


	opt = RMSprop(lr=0.0002, decay=1e-6)

	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
	
	
	
	
	
	
	
	
	print ('Training...')
	model.fit(x, y, batch_size=BATCH_SIZE, nb_epoch=EPOCH_COUNT, verbose=1)

	return model




fname = 'kapi_resampled.wav'
data = get_ffts(fname)
model = train_model(data)


#print(np.shape(x))
#print(np.shape(y))





'''
plt.matshow(data)
plt.show()
'''

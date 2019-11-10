import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils import to_categorical
#import scikit-learn
import pandas as pd
import os

#define
#X_train
#X_test
def inverse_dummies(data, categories):
	column = np.argmax(data,axis=1)
	print(column[:10])
	a = list()
	for i in range(len(column)):
		a.append(categories[column[i]])
	return a

    
LAYER_NUM = 5
HIDDEN_DIM = 64
look_back = 20


data_dir = 'data.txt'
#create one_hot vectors
#epochs
raw_data = open(data_dir,'r').read()
data = list((raw_data.split('\n')))
data =  [int(x) for x in data[:100]]


freqs = list(set(data))
freqs = [int(x) for x in freqs]


freqs.sort()
#print(freqs[34])
vocab_size = len(freqs)
print(freqs)
print(vocab_size)


#Converting to enum

dataset = pd.get_dummies(data)
#print(dataset)
dataset = dataset.as_matrix()
#print(dataset)
#print(np.argmax(dataset,1))
#x = inverse_dummies(dataset,freqs)
#print(x[:20])
#dataset = to_categorical(data)
#print(np.argmax(dummies))
#print(np.argmax(encoded_data[:6][:end]))

#print(dataset)
#seq = encoded_data

#neural netwrtk
#num = enum(freqs)
#print(num)
#need to change this



# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),:]
		dataX.append(a)
		dataY.append(dataset[i + look_back,:])
	return np.array(dataX), np.array(dataY)
# fix random seed for reproducibilityfff
np.random.seed(7)
# load the dataset
#dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#dataset = dataframe.values
#dataset = dataset.astype('float32')
# normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset))
#test_size = len(dataset) - train_size
train= dataset
print(np.shape(train))
# reshape into X=t and Y=t+1

trainX, trainY = create_dataset(train, look_back)
#testX, testY = create_dataset(test, look_back)
print(np.shape(trainX))
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back,trainX.shape[2]))
trainY = np.reshape(trainY, (trainX.shape[0], 1,trainX.shape[2]))
print(trainX[-1][:][:])
#testX = np.reshape(testX, (testX.shape[0], look_back,testX.shape[2]))
print(np.shape(trainX))
print(np.shape(trainY))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None,vocab_size),return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size)))
model.add(TimeDistributed(Dropout(0.2)))
#model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

training_data = np.save('training_datafull.npy', trainX)


model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)


def extend(train, last_note,look_back):
	full = np.vstack((train,last_note))
	return full[:][1:look_back+1][:]



store = open('a.txt', 'w')
store.close()
n = 0
model.save('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, n))
trainX = trainX[-1][:][:]
trainX = np.reshape(trainX, (1, look_back,vocab_size))
while True:
	store = open('a.txt', 'a')
	print('\n')
	#epochs += 1
	n += 1
	print(np.shape(trainX))
	out = model.predict(trainX)
	print(np.shape(out))
	print(out)
	out_bin = to_categorical(np.argmax(out),vocab_size)
	pred =np.argmax(out)
	print(pred)
	#print(pred)
	print(out_bin)
	print(np.shape(out_bin))
	last_note = np.reshape(out_bin, (trainX.shape[0], 1,trainX.shape[2]))
	trainX = extend(trainX,last_note,look_back)
	generated = np.reshape(out_bin,(1,vocab_size))
	x = inverse_dummies(generated,freqs)
	print(x)
	store.write('{}\n'.format(x))
	store.close()
	#model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=1) 
	if n % 10 == 0:
		model.save('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, n))


# make predictions
#trainPredict = model.predict(trainX)
#testPredict = model.predict(testX)
# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

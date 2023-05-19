#!/usr/bin/env python3
##########################################################################################################################
##
## The airline passengers problem solved with a LSTM
##
## Represents Problem 1 from https://www.frontiersin.org/articles/10.3389/fncom.2021.705050/full
## and corresponding MSc thesis
## https://nur.nu.edu.kz/bitstream/handle/123456789/3789/ThesisFinal_KazybekAdam_Library.pdf?sequence=1&isAllowed=y
##
## Initial code from
## https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
##
##########################################################################################################################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# import norm
from keras.constraints import min_max_norm

# dataset = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python')
# plt.plot(dataset)
# plt.show()

# fix random seed for reproducibility
tf.random.set_seed(7)

# load the dataset
dataframe = pd.read_csv("airline.csv", usecols=[1], engine="python")
dataset = dataframe.values.astype("float32")

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size : len(dataset), :]
print(len(train), len(test))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# reshape into X=t and Y=t+1
look_back = 2
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(
    LSTM(
        4,
        input_shape=(1, look_back),
        kernel_constraint=min_max_norm(-1, 1),
        recurrent_constraint=min_max_norm(-1, 1),
        bias_constraint=min_max_norm(-1.1),
    )
)
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(trainX, trainY, epochs=500, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print("Train Score: %.2f RMSE" % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print("Test Score: %.2f RMSE" % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back : len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[
    len(trainPredict) + (look_back * 2) + 1 : len(dataset) - 1, :
] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), label="entire dataset")
plt.plot(trainPredictPlot, label="trainPredict")
plt.plot(testPredictPlot, label="testPredict")
plt.legend(loc="best")
plt.show()

# from https://stackoverflow.com/questions/42861460/how-to-interpret-weights-in-a-lstm-layer-in-keras
weights_trainable = model.layers[0].trainable_weights
# print("\nTrainable weights:")
# print(weights_trainable)

units = int(int(model.layers[0].trainable_weights[0].shape[1]) / 4)
print("\nNo units: ", units)

## Each tensor contains weights for four LSTM units (in this order): i (input), f (forget), c (cell state) and o (output)
W = model.layers[0].get_weights()[0]
U = model.layers[0].get_weights()[1]
b = model.layers[0].get_weights()[2]

print("\nShape of W, should be (", look_back, " x ", 4 * units, "): ", W.shape)
print("\nShape of U, should be (", units, " x ", 4 * units, "): ", U.shape)
print("\nShape of b should be (", 4 * units, "): ", b.shape)

W_xi = W[:, :units]
W_xf = W[:, units : units * 2]
W_xc = W[:, units * 2 : units * 3]
W_xo = W[:, units * 3 :]

U_hi = U[:, :units]
U_hf = U[:, units : units * 2]
U_hc = U[:, units * 2 : units * 3]
U_ho = U[:, units * 3 :]

b_i = b[:units]
b_f = b[units : units * 2]
b_c = b[units * 2 : units * 3]
b_o = b[units * 3 :]

print("\nW_o:")
print(W_xo)
print("\nW_i:")
print(W_xi)
print("\nW_c:")
print(W_xc)
print("\nW_f:")
print(W_xf)
print("\nU_o:")
print(U_ho)
print("\nU_i:")
print(U_hi)
print("\nU_c:")
print(U_hc)
print("\nU_f:")
print(U_hf)
print("\nb_o:")
print(b_o)
print("\nb_i:")
print(b_i)
print("\nb_c:")
print(b_c)
print("\nb_f:")
print(b_f)


model.summary()

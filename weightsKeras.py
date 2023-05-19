#!/usr/bin/env python3

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


def create_dataset(ds, lookback=1):
    X, y = [], []
    for i in range(len(ds) - lookback):
        feature = ds[i : i + lookback, 0]
        target = ds[i + lookback, 0]
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)


# LSTMs have unique 3-dimensional input requirements

df = pd.read_csv("airline.csv")
ds = df[["Passengers"]].values.astype("float32")

# normalize the ds
scaler = MinMaxScaler(feature_range=(0, 1))
ds = scaler.fit_transform(ds)
print(ds)

# split into train and test sets
train_size = int(len(ds) * 0.67)
test_size = len(ds) - train_size
train, test = ds[0:train_size, :], ds[train_size : len(ds), :]
print(len(train), len(test))

# reshape into X=t and Y=t+1
lookback = 2
trainX, trainY = create_dataset(train, lookback)
testX, testY = create_dataset(test, lookback)

print(trainX.shape, trainX.shape)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(
    LSTM(
        4,
        input_shape=(1, lookback),
        kernel_constraint=min_max_norm(-1, 1),
        recurrent_constraint=min_max_norm(-1, 1),
        bias_constraint=min_max_norm(-1, 1),
    )
)

# model.add(Dense(2,kernel_initializer='normal',activation='linear'))
model.add(Dense(1, kernel_initializer="normal", activation="linear"))
print(model.summary())
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(trainX, trainY, epochs=500, batch_size=1, validation_split=0.35, verbose=1)
scores = model.evaluate(trainX, trainY, verbose=1, batch_size=1)
print("Accurracy: {}".format(scores[1]))


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
predict = model.predict(X)

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

# plt.plot(y, predict-y, 'C2')
# plt.ylim(ymax = 3, ymin = -3)
plt.plot(predict)
plt.show()

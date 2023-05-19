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

# Global vars
lookback = 2


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
X = np.zeros((ds.shape[0] - lookback + 1, lookback))

for i in range(len(X)):
    X[i][0] = ds[i][0]
    for j in range(1, lookback):
        X[i][j] = ds[i + j]
        # X[i] = np.append(ds[i], ds[i + 1 : i + lookback][0], 0)

# split into train and test sets
train_size = int(len(ds) * 0.67)
test_size = len(ds) - train_size
train, test = ds[0:train_size, :], ds[train_size : len(ds), :]

# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train, lookback)
testX, testY = create_dataset(test, lookback)


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(
    LSTM(
        4,
        input_shape=(lookback, 1),
        kernel_constraint=min_max_norm(-1, 1),
        recurrent_constraint=min_max_norm(-1, 1),
        bias_constraint=min_max_norm(-1, 1),
    )
)

# model.add(Dense(2, kernel_initializer="normal", activation="linear"))
model.add(Dense(1, kernel_initializer="normal", activation="linear"))
print(model.summary())
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(trainX, trainY, epochs=30, batch_size=1, validation_split=0.35, verbose=1)
scores = model.evaluate(trainX, trainY, verbose=1, batch_size=1)
print("Accurracy: {}".format(scores[1]))


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
predict = model.predict(X)
predict = scaler.inverse_transform(predict)

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


trainPredict, testPredict = (
    predict[0 : train_size - lookback, :],
    predict[train_size - lookback : len(predict), :],
)
# print(len(trainPredict), len(testPredict), len(predict), len(ds), trainY.shape)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(ds)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback - 1 : len(trainPredict) + lookback - 1, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(ds)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + lookback - 2 : len(ds) - 1, :] = testPredict

plt.plot(scaler.inverse_transform(ds), label="entire dataset")
plt.plot(trainPredictPlot, label="trainPredict")
plt.plot(testPredictPlot, label="testPredict")
plt.legend(loc="best")
plt.show()

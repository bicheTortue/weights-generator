#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# import norm
from tensorflow.keras.constraints import MinMaxNorm

from keras import backend as K
from tensorflow.keras.layers import Activation

from nn_backend import *

# Global vars
lookback = 2
nbInput = 1
nbHidden = 4
nbOutput = 1
epochs = 300


def main():
    # LSTMs have unique 3-dimensional input requirements

    tf.random.set_seed(7)
    df = pd.read_csv("airline.csv")
    ds = df[["Passengers"]].values.astype("float32")

    # normalize the ds
    scaler = MinMaxScaler(feature_range=(0, 1))
    ds = scaler.fit_transform(ds)

    # split into train and test sets
    train_size = int(len(ds) * 0.67)
    test_size = len(ds) - train_size
    train, test = ds[0:train_size, :], ds[train_size : len(ds), :]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, lookback)
    testX, testY = create_dataset(test, lookback)
    X, _ = create_dataset(ds, lookback)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], nbInput))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], nbInput))
    X = np.reshape(X, (X.shape[0], X.shape[1], nbInput))

    # create and fit the LSTM network
    model = Sequential()
    model.add(
        LSTM(
            nbHidden,
            input_shape=(lookback, nbInput),
            kernel_constraint=MinMaxNorm(-1, 1),
            recurrent_constraint=MinMaxNorm(-1, 1),
            bias_constraint=MinMaxNorm(-1, 1),
            recurrent_activation=cSigmoid(),
            activation=cTanh(),
        )
    )
    # model.add(Dense(2, kernel_initializer="normal", activation="linear"))
    model.add(Dense(nbOutput, kernel_initializer="normal", activation="linear"))

    print(model.summary())

    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    model.fit(
        trainX, trainY, epochs=epochs, batch_size=1, validation_split=0.35, verbose=1
    )
    scores = model.evaluate(trainX, trainY, verbose=1, batch_size=1)
    print("Accurracy: {}".format(scores[1]))

    model.save("airline.keras")


if __name__ == "__main__":
    main()

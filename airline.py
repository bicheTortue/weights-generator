#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

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


def create_model():
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
    model.add(Dense(nbOutput, kernel_initializer="normal", activation="linear"))
    return model


def get_dataset():
    df = pd.read_csv("airline.csv")
    ds = df[["Passengers"]].values.astype("float32")

    return ds


def train():
    # LSTMs have unique 3-dimensional input requirements
    tf.random.set_seed(7)

    ds = get_dataset()

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

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], nbInput))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], nbInput))

    model = create_model()

    print(model.summary())

    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    model.fit(
        trainX, trainY, epochs=epochs, batch_size=1, validation_split=0.35, verbose=1
    )
    scores = model.evaluate(trainX, trainY, verbose=1, batch_size=1)
    print("Accurracy: {}".format(scores[1]))

    saveTofile(model.layers, "airline.wei")
    model.save_weights("airline.keras")


def pred():
    model = create_model()

    print(model.summary())
    model.load_weights("airline.keras")

    ds = get_dataset()
    # normalize the ds
    scaler = MinMaxScaler(feature_range=(0, 1))
    ds = scaler.fit_transform(ds)

    X, _ = create_dataset(ds, lookback)
    X = np.reshape(X, (X.shape[0], X.shape[1], nbInput))

    # make predictions
    predict = model.predict(X)
    df = pd.DataFrame(predict)
    df.columns = ["digital"]
    df.to_csv("predict.csv")
    predict = scaler.inverse_transform(predict)

    # Separate the train and test
    trainPredict, testPredict = (
        predict[0 : train_size + 1, :],
        predict[train_size : len(predict), :],
    )

    # calculate root mean squared error
    # trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    # print("Train Score: %.2f RMSE" % (trainScore))
    # testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    # print("Test Score: %.2f RMSE" % (testScore))

    # print(len(trainPredict), len(testPredict), len(predict), len(ds), trainY.shape)

    # shift train predictions for plotting
    trainPredictPlot = np.ones_like(ds) * np.nan
    trainPredictPlot[lookback : len(trainPredict) + lookback, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.ones_like(ds) * np.nan
    testPredictPlot[len(trainPredict) + 1 : len(ds), :] = testPredict

    plt.plot(scaler.inverse_transform(ds), label="entire dataset")
    plt.plot(predict, label="predict")
    # plt.plot(trainPredictPlot, label="trainPredict")
    # plt.plot(testPredictPlot, label="testPredict")
    plt.legend(loc="best")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="LSTM weight",
        description="This program is used to generate the weights used to later import in a netlist.",
    )
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pred", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    if args.train:
        train()

    if args.pred:
        pred()


if __name__ == "__main__":
    main()

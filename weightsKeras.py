#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# from tensorflow.keras.utils.generic_utils import get_custom_objects

# Note! You cannot use random python functions, activation function gets as an input tensorflow tensors and should return tensors. There are a lot of helper functions in keras backend.


class cTanh(Layer):
    def __init__(self, beta, **kwargs):
        super(cTanh, self).__init__(**kwargs)
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        # return K.tanh(self.beta * inputs)
        return K.sigmoid(self.beta * inputs) * 2 - 1

    def get_config(self):
        config = {"beta": float(self.beta)}
        base_config = super(cTanh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class cSigm(Layer):
    def __init__(self, beta, **kwargs):
        super(cSigm, self).__init__(**kwargs)
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return K.sigmoid(self.beta * inputs)

    def get_config(self):
        config = {"beta": float(self.beta)}
        base_config = super(cSigm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# c is for custom
@tf.function
def cSigmoid(x):
    l = np.loadtxt(
        "sigmoid.csv", delimiter=",", dtype=np.float32
    )  # Value need to be transformed
    out = []
    print(x)
    print(x[None, 1, 0])
    for i in x:
        print(i)
        for j in i:
            print(j)
    for i in range(len(l) - 1):
        # if K.cast(K.greater(l[i + 1, 0], x)):
        if x[0][0] < l[i + 1, 0]:
            a = (l[i + 1, 1] - l[i, 1]) / (l[i + 1, 0] - l[i, 0])
            b = l[i, 1] - a * l[i, 0]
            out.append(x * a + b)
    return K.constant(out)

    # return 1 / (1 + K.exp(-x))


# get_custom_objects().update({'custom_activation': Activation(custom_activation)})


# Global vars
lookback = 2
nbInput = 1
nbHidden = 15
nbOutput = 1
epochs = 150


def create_dataset(ds, lookback=1):
    X, y = [], []
    for i in range(len(ds) - lookback):
        feature = ds[i : i + lookback, 0]
        target = ds[i + lookback, 0]
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)


def getLSTMWeights(lstm, nbInput, nbHidden):
    W = lstm.get_weights()[0]
    U = lstm.get_weights()[1]
    b = lstm.get_weights()[2]
    nbGates = 4
    out = [None] * nbGates

    for i in range(nbGates):
        out[i] = []
        for j in range(nbHidden):
            for k in range(nbInput):
                out[i].extend(W[:, i * nbGates + j * nbInput])
            out[i].extend(U[:, i])
            out[i].append(b[i * nbGates + j])
    return out


def getLinearWeights(nn, nbOutput):
    W = nn.get_weights()[0]
    b = list(nn.get_weights()[1])
    out = []
    for i in range(nbOutput):
        out.extend(W[:, i])
        out.append(b[i])
    return out


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
            # recurrent_activation=Activation(cSigmoid),
            recurrent_activation=cSigm(0.67),
            activation=cTanh(0.67),
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
    plt.plot(trainPredictPlot, label="trainPredict")
    plt.plot(testPredictPlot, label="testPredict")
    plt.legend(loc="best")
    plt.show()
    np.savetxt("lstm.wei", getLSTMWeights(model.layers[0], nbInput, nbHidden))
    np.savetxt("dense.wei", getLinearWeights(model.layers[1], nbOutput))


if __name__ == "__main__":
    main()

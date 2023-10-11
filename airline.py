#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from keras.models import load_model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

# import norm
from tensorflow.keras.constraints import MinMaxNorm

from keras import backend as K
from tensorflow.keras.layers import Activation

from barbalib import *


def create_model(args):
    if args.custom:
        sigm = cSigmoid()
        tanh = cTanh()
    else:
        sigm = Activation("sigmoid")
        tanh = Activation("tanh")
    model = Sequential()
    # Limits for the weights in the lstm
    lim_val = 9 / (args.input_size + args.hidden_size + 1)
    limits = MinMaxNorm(-lim_val, lim_val)
    model.add(
        LSTM(
            args.hidden_size,
            input_shape=(args.lookback, args.input_size),
            kernel_constraint=limits,
            recurrent_constraint=limits,
            bias_constraint=limits,
            recurrent_activation=sigm,
            activation=tanh,
        )
    )
    model.add(
        Dense(
            args.output_size,
            kernel_initializer="normal",
            activation="linear",
            # kernel_constraint=limits,
            # bias_constraint=limits,
        )
    )
    return model


def get_dataset():
    df = pd.read_csv("airline.csv")
    ds = df[["Passengers"]].values.astype("float32")

    return ds


def create_dataset(ds, lookback=1):
    X, y = [], []
    for i in range(len(ds) - lookback):
        feature = ds[i: i + lookback, 0]
        target = ds[i + lookback, 0]
        X.append(feature)
        y.append(target)


tt return np.array(X), np.array(y)


def train(args):
    # LSTMs have unique 3-dimensional input requirements
    tf.random.set_seed(7)

    ds = get_dataset()

    # normalize the ds
    scaler = MinMaxScaler(feature_range=(0, 1))
    ds = scaler.fit_transform(ds)

    # split into train and test sets
    train_size = int(len(ds) * 0.67)
    test_size = len(ds) - train_size
    train, test = ds[0:train_size, :], ds[train_size: len(ds), :]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, args.lookback)
    testX, testY = create_dataset(test, args.lookback)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(
        trainX, (trainX.shape[0], trainX.shape[1], args.input_size))
    testX = np.reshape(
        testX, (testX.shape[0], testX.shape[1], args.input_size))

    model = create_model(args)

    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    model.fit(
        trainX,
        trainY,
        epochs=args.epochs,
        batch_size=1,
        validation_split=0.35,
        verbose=1,
    )
    model.summary()
    scores = model.evaluate(trainX, trainY, verbose=1, batch_size=1)
    print("Accurracy: {}".format(scores[1]))

    saveTofile(model.layers, "airline.wei")
    # model.save_weights("airline.keras")
    model.save("airline.h5")


def pred(args):
    model = load_model(
        "airline.h5",
        custom_objects={"cTanh": cTanh,
                        "cSigmoid": cSigmoid, "Activation": Activation},
    )

    model.summary()

    ds = get_dataset()
    # normalize the ds
    scaler = MinMaxScaler(feature_range=(0, 1))
    ds = scaler.fit_transform(ds)

    X, _ = create_dataset(ds, args.lookback)
    X = np.reshape(X, (X.shape[0], X.shape[1], args.input_size))

    # make predictions
    predict = model.predict(X)
    df = pd.DataFrame(predict)
    df.columns = ["digital"]
    if args.save:
        df.to_csv("predict.csv")
    if args.plot:
        predict = scaler.inverse_transform(predict)
        plt.plot(scaler.inverse_transform(ds), label="entire dataset")
        plt.plot(predict, label="predict")
        plt.legend(loc="best")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="LSTM weight",
        description="This program is used to generate the weights used to later import in a netlist.",
    )
    parser.add_argument(
        "--train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--pred", action=argparse.BooleanOptionalAction, default=False)

    # Global args
    parser.add_argument(
        "--model", choices=["LSTM", "GRU", "RNN"], default="LSTM")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lookback", type=int, default=2)
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=4)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    # Train required
    parser.add_argument(
        "--custom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the training will use the custom sigmoid and tanh functions",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "rmsprop", "adam"],
        default="adam",
    )

    # Pred required
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    if args.train:
        train(args)

    if args.pred:
        pred(args)


if __name__ == "__main__":
    main()

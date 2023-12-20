#!/usr/bin/env python3

from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import glob
from keras.models import load_model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from tensorflow.keras.layers import Layer
from sklearn.preprocessing import MinMaxScaler

# import norm
from tensorflow.keras.constraints import MinMaxNorm

from plotNread import *
from barbalib import *


def create_model(opt, output_size):
    if opt.custom:
        sigm = cSigmoid()
        tanh = cTanh()
    else:
        sigm = Activation("sigmoid")
        tanh = Activation("tanh")
    lim_val = 9 / (opt.input_size + opt.hidden_size + 1)
    limits = MinMaxNorm(-lim_val, lim_val)
    model = Sequential()
    if opt.model == "LSTM":
        model.add(
            LSTM(
                opt.hidden_size,
                input_shape=(1000, 4),
                kernel_constraint=limits,
                recurrent_constraint=limits,
                bias_constraint=limits,
                recurrent_activation=sigm,
                activation=tanh,
                return_sequences=True,
            )
        )
    elif opt.model == "GRU":
        model.add(
            GRU(
                opt.hidden_size,
                recurrent_activation=cSigmoid(),
                activation=cTanh(),
                return_sequences=True,
            )
        )
    elif opt.model == "RNN":
        model.add(
            SimpleRNN(
                opt.hidden_size,
                recurrent_activation=cSigmoid(),
                activation=cTanh(),
                return_sequences=True,
            )
        )
    model.add(TimeDistributed(Dense(output_size)))

    return model


def save_history(history, opt):
    dat = np.array([history.history["loss"], history.history["val_loss"]])
    a = np.column_stack((dat))
    np.savetxt(opt.savepath + "training_history.dat", a, delimiter=" ")


def load_best_weights(opt, model):
    checkpoint_dir = opt.savepath + "weights/*"
    list_files = glob.glob(checkpoint_dir)
    latest = max(list_files, key=os.path.getctime)
    # print(latest)
    model.load_weights(latest)

    return model


def get_callbacks(opt):
    filepath = opt.savepath + "weights/weights-{epoch:02d}-{val_loss:.6f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )
    if opt.early_stop == False:
        callbacks_list = [checkpoint]
    elif opt.early_stop == True:
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=opt.patience,
            verbose=1,
            mode="auto",
        )
        callbacks_list = [checkpoint, early_stop]

    return callbacks_list


def evaluate_results(
    model, opt, trainx, trainy, validx, validy, testx, testy, output_size
):
    # Evaluate on training, validation and test data

    train_l = model.evaluate(
        np.array(trainx),
        np.array(trainy),
        batch_size=opt.batch_size,
        verbose=1,
        workers=12,
        use_multiprocessing=True,
    )
    valid_l = model.evaluate(
        np.array(validx),
        np.array(validy),
        batch_size=opt.batch_size,
        verbose=1,
        workers=12,
        use_multiprocessing=True,
    )
    test_l = model.evaluate(
        np.array(testx),
        np.array(testy),
        batch_size=opt.batch_size,
        verbose=1,
        workers=12,
        use_multiprocessing=True,
    )

    t1 = time()
    trainPredict = model.predict(
        np.array(trainx),
        batch_size=opt.batch_size,
        verbose=1,
        workers=12,
        use_multiprocessing=True,
    )
    plot_results(opt, trainy, trainPredict, "/train", output_size)
    validPredict = model.predict(
        np.array(validx),
        batch_size=opt.batch_size,
        verbose=1,
        workers=12,
        use_multiprocessing=True,
    )
    plot_results(opt, validy, validPredict, "/valid", output_size)
    testPredict = model.predict(
        np.array(testx),
        batch_size=opt.batch_size,
        verbose=1,
        workers=12,
        use_multiprocessing=True,
    )
    t2 = time()
    print("Took :", t2 - t1, "seconds")
    plot_results(opt, testy, testPredict, "/test", output_size)

    return train_l, valid_l, test_l


def train(opt):
    ###########################################################################
    # Variables Definition
    ###########################################################################
    nin = ["time", "PLML2", "PLMR", "AVBL", "AVBR"]
    nout = ["time", "DB1", "LUAL", "PVR", "VB1"]
    neurons = ["time", "DB1", "LUAL", "PVR", "VB1", "PLML2", "PLMR", "AVBL", "AVBR"]
    if opt.optimizer == "adam":
        opt_function = Adam(learning_rate=opt.learning_rate)
    elif opt.optimizer == "sgd":
        opt_function = SGD(learning_rate=opt.learning_rate)
    elif opt.optimizer == "rmsprop":
        opt_function = RMSprop(learning_rate=opt.learning_rate)

    ###########################################################################
    # Read data
    ###########################################################################
    files = get_data(opt.datapath, opt.extension)
    train, valid, test = split_data(files)
    trainx, trainy = read_data(opt.datapath, train, neurons, nin, nout)
    validx, validy = read_data(opt.datapath, valid, neurons, nin, nout)
    testx, testy = read_data(opt.datapath, test, neurons, nin, nout)
    if opt.plots:
        plot_data(trainx, "/train_data", "/x", opt.model, opt.savepath)
        plot_data(trainy, "/train_data", "/y", opt.model, opt.savepath)
        plot_data(validx, "/valid_data", "/x", opt.model, opt.savepath)
        plot_data(validy, "/valid_data", "/y", opt.model, opt.savepath)
        plot_data(testx, "/test_data", "/x", opt.model, opt.savepath)
        plot_data(testy, "/test_data", "/y", opt.model, opt.savepath)
    ###########################################################################
    # Create and Train Model
    ###########################################################################
    output_size = 4
    model = create_model(opt, output_size)
    model.compile(optimizer=opt_function, loss="mean_squared_error")
    callbacks_list = get_callbacks(opt)
    history = model.fit(
        np.array(trainx),
        np.array(trainy),
        batch_size=opt.batch_size,
        epochs=opt.epochs,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(np.array(validx), np.array(validy)),
        validation_freq=1,
        workers=12,
        use_multiprocessing=True,
    )
    model.summary()
    save_history(history, opt)
    model = load_best_weights(opt, model)
    if opt.save:
        saveTofile(model.layers, "celegans.wei")
    model.save(opt.savepath + "celegans.h5")


def pred(opt):
    ###########################################################################
    # Variables Definition
    ###########################################################################
    nin = ["time", "PLML2", "PLMR", "AVBL", "AVBR"]
    nout = ["time", "DB1", "LUAL", "PVR", "VB1"]
    neurons = ["time", "DB1", "LUAL", "PVR", "VB1", "PLML2", "PLMR", "AVBL", "AVBR"]

    ###########################################################################
    # Read data
    ###########################################################################
    files = get_data(opt.datapath, opt.extension)
    train, valid, test = split_data(files)
    trainx, trainy = read_data(opt.datapath, train, neurons, nin, nout)
    validx, validy = read_data(opt.datapath, valid, neurons, nin, nout)
    testx, testy = read_data(opt.datapath, test, neurons, nin, nout)
    if opt.plots_in:
        plot_data(trainx, "/train_data", "/x", opt.model, opt.savepath)
        plot_data(trainy, "/train_data", "/y", opt.model, opt.savepath)
        plot_data(validx, "/valid_data", "/x", opt.model, opt.savepath)
        plot_data(validy, "/valid_data", "/y", opt.model, opt.savepath)
        plot_data(testx, "/test_data", "/x", opt.model, opt.savepath)
        plot_data(testy, "/test_data", "/y", opt.model, opt.savepath)
    ###########################################################################
    # Load Model and Evaluate
    ###########################################################################
    output_size = 4
    # model = create_model(opt, output_size)
    # model.load_weights("celegans.keras")
    model = load_model(
        opt.savepath + "celegans.h5",
        custom_objects={"cTanh": cTanh, "cSigmoid": cSigmoid, "Activation": Activation},
    )
    model.summary()

    lt, lv, ltt = evaluate_results(
        model, opt, trainx, trainy, validx, validy, testx, testy, output_size
    )
    print("Training Loss: " + str(lt))
    print("Validation Loss: " + str(lv))
    print("Test Loss: " + str(ltt))


def main():
    parser = argparse.ArgumentParser(
        prog="LSTM weight",
        description="This program is used to generate the weights used to later import in a netlist.",
    )
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pred", action=argparse.BooleanOptionalAction, default=False)

    ###########################################################################
    # Parser Definition
    ###########################################################################
    # Global args
    parser.add_argument("--model", choices=["LSTM", "GRU", "RNN"], default="LSTM")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--datapath", type=str, default="./celegans_data/")
    parser.add_argument("--savepath", type=str, default="./celegans_out/run0/")
    parser.add_argument("--extension", type=str, default=".dat")
    parser.add_argument("--batch_size", type=int, default=32)
    # Train required
    parser.add_argument(
        "--custom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the training will use the custom sigmoid and tanh functions",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--input_size", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "rmsprop", "adam"],
        default="adam",
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--early_stop", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--patience", type=int, default=50)

    # Pred required
    parser.add_argument(
        "--plots_in", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--plots_out", action=argparse.BooleanOptionalAction, default=True
    )

    args = parser.parse_args()

    if args.train:
        train(args)

    if args.pred:
        pred(args)


if __name__ == "__main__":
    main()

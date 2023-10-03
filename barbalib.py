import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


class cSigmoid(Layer):
    def __init__(self, **kwargs):
        super(cSigmoid, self).__init__(**kwargs)

    def call(self, inputs):
        l = np.loadtxt("sigmoid.csv", delimiter=",", dtype=np.float32)
        l[:, 0] = (l[:, 0]) * 10
        l[:, 1] = (l[:, 1] - 0.9) * 10

        # cond = [tf.cast(tf.math.less(inputs, l[i, 0]), tf.float32)]

        f = tf.constant(0.0)
        for i in range(len(l) - 1):
            cond = tf.cast(
                tf.math.logical_and(
                    tf.math.greater_equal(inputs, l[i, 0]),
                    tf.math.less(inputs, l[i + 1, 0]),
                ),
                tf.float32,
            )
            # cond = tf.less(inputs, l[i + 1, 0])
            a = (l[i + 1, 1] - l[i, 1]) / (l[i + 1, 0] - l[i, 0])
            b = l[i, 1] - a * l[i, 0]
            f += tf.math.multiply(cond, a * inputs + b)

        cond = tf.cast(tf.math.greater_equal(inputs, l[i, 0]), tf.float32)
        f += tf.math.multiply(cond, tf.constant(1.0))

        return f

    def get_config(self):
        # config = {"beta": float(self.beta)}
        base_config = super(cSigmoid, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class cTanh(Layer):
    def __init__(self, **kwargs):
        super(cTanh, self).__init__(**kwargs)

    def call(self, inputs):
        l = np.loadtxt("tanh.csv", delimiter=",", dtype=np.float32)
        l[:, 0] = (l[:, 0]) * 10
        l[:, 1] = (l[:, 1] - 0.9) * 10

        # cond = [tf.cast(tf.math.less(inputs, l[i, 0]), tf.float32)]

        f = tf.constant(0.0)
        cond = tf.cast(tf.math.less(inputs, l[0, 0]), tf.float32)
        f += tf.math.multiply(cond, tf.constant(-1.0))

        for i in range(len(l) - 1):
            cond = tf.cast(
                tf.math.logical_and(
                    tf.math.greater_equal(inputs, l[i, 0]),
                    tf.math.less(inputs, l[i + 1, 0]),
                ),
                tf.float32,
            )
            # cond = tf.less(inputs, l[i + 1, 0])
            a = (l[i + 1, 1] - l[i, 1]) / (l[i + 1, 0] - l[i, 0])
            b = l[i, 1] - a * l[i, 0]
            f += tf.math.multiply(cond, a * inputs + b)

        cond = tf.cast(tf.math.greater_equal(inputs, l[i, 0]), tf.float32)
        f += tf.math.multiply(cond, tf.constant(1.0))

        return f

    def get_config(self):
        base_config = super(cTanh, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def create_dataset(ds, lookback=1):
    X, y = [], []
    for i in range(len(ds) - lookback):
        feature = ds[i : i + lookback, 0]
        target = ds[i + lookback, 0]
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)


def getLSTMWeights(lstm, nbHidden):
    W = lstm.get_weights()[0]
    U = lstm.get_weights()[1]
    b = lstm.get_weights()[2]
    nbGates = 4
    out = [None] * nbGates

    for i in range(nbGates):
        out[i] = []
        for j in range(nbHidden):
            out[i].extend(W[:, i * nbGates + j])
            out[i].extend(U[:, i * nbGates + j])
            out[i].append(b[i * nbGates + j])
    return out


def getDenseWeights(nn, nbOutput):
    W = nn.get_weights()[0]
    b = list(nn.get_weights()[1])
    out = []
    for i in range(nbOutput):
        out.extend(W[:, i])
        out.append(b[i])
    return out


def saveTofile(layers, filename):
    out = [[]]
    for layer in layers:
        if type(layer) == Dense:
            nbOut = layer.output_shape[1]
            out[0].append("Dense(" + str(nbOut) + ")")
            out.append(getDenseWeights(layer, nbOut))
        elif type(layer) == LSTM:
            nbHid = layer.units
            out[0].append("LSTM(" + str(nbHid) + ")")
            out.append(getLSTMWeights(layer, nbHid))

    with open(filename, "wb") as file:  # Pickling
        pickle.dump(out, file)

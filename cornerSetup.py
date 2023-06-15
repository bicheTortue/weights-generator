#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Global vars
lookback = 2


def create_dataset(ds, lookback=1):
    X, y = [], []
    for i in range(len(ds) - lookback):
        feature = ds[i: i + lookback, 0]
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

X, _ = create_dataset(ds, lookback)

# Does not support more than 1 input so far

for i, x in enumerate(X):
    print("<corner enabled='1'>test" + str(i))
    print("<vars>")
    for j, val in enumerate(x):
        # nb inputs will have to change here (the zero)
        print("<var>in0step" + str(j))
        print("<value>" + str(float('%.2g' % val / 10)) + "</value>")
        print("</var>")
    print("<var>step")
    print("<value>" + str(i) + "</value>")
    print("</var>")
    print("</vars>")
    print("</corner>")

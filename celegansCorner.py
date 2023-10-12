#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from plot_read import *


nin = ["time", "PLML2", "PLMR", "AVBL", "AVBR"]
nout = ["time", "DB1", "LUAL", "PVR", "VB1"]
neurons = ["time", "DB1", "LUAL", "PVR", "VB1", "PLML2", "PLMR", "AVBL", "AVBR"]

files = get_data("celegans_data/", ".dat")
X, _ = read_data("celegans_data/", files, neurons, nin, nout)

print(X[0].shape)
# for i in X[0].iloc[:, 0]:
# print(i)

for i, x in enumerate(X):
    print("<corner enabled='1'>test" + str(i))
    print("<vars>")
    for j in range(x.shape[-1]):
        for k, val in enumerate(x.iloc[:, j]):
            # nb inputs will have to change here (the zero)
            print("<var>in" + str(j) + "step" + str(k))
            tmpVal = val / 10
            tmpVal = float("%.2g" % tmpVal)
            print("<value>" + str(tmpVal) + "</value>")
            print("</var>")
    print("</vars>")
    print("</corner>")

#!/usr/bin/env python3

# Load Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation  # Generate 2 sets of X variables

# LSTMs have unique 3-dimensional input requirements

df = pd.read_csv("airline.csv")
df2 = np.array(df["Passengers"])
X = df2[:-1]
X = X.reshape((143, 1))
Y = df2[1:]
Y = Y.reshape((143, 1))

model = Sequential()
model.add(
    LSTM(50, input_shape=(None, 1), return_sequences=False)
)  # True = many to many
# model.add(Dense(2,kernel_initializer='normal',activation='linear'))
model.add(Dense(1, kernel_initializer="normal", activation="linear"))
print(model.summary())
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(X, Y, epochs=2000, batch_size=1, validation_split=0.35, verbose=1)
scores = model.evaluate(X, Y, verbose=1, batch_size=1)
print("Accurracy: {}".format(scores[1]))

predict = model.predict(X)
print(predict)
plt.plot(predict)

# plt.plot(y, predict-y, 'C2')
# plt.ylim(ymax = 3, ymin = -3)
plt.show()

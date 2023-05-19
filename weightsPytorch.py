#!/usr/bin/env python3

import numpy as np
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AirModel(nn.Module):
    def __init__(self, nbHidden):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=nbHidden, batch_first=True)
        self.linear = nn.Linear(nbHidden, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        # x = x[:, -1, :]
        x = self.linear(x)
        return x


class AirModel2(nn.Module):
    def __init__(self, nbInput, nbHidden):
        super().__init__()
        self.lstm = nn.LSTM(input_size=nbInput, hidden_size=nbHidden, batch_first=True)
        self.linear1 = nn.Linear(nbHidden, 2)
        self.linear2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        # x = x[:, -1, :]
        x = self.linear1(x)
        x = self.linear2(x)
        # x = self.sigmoid(x)
        return x


def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i : i + lookback]
        target = dataset[i + 1 : i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


def getLSTMWeights(lstm, nbInput, nbHidden):
    nbGates = 4
    weights = list(lstm.parameters())
    out = [None] * nbGates
    for i in range(4):
        weights[i] = weights[i].detach().numpy()

    weights[2] = weights[2] + weights[3]
    weights.pop()

    print(weights)
    print("=============\n")
    for i in range(nbGates):
        out[i] = []
        for j in range(nbHidden):
            out[i].extend(weights[0][i * nbHidden + j])
            out[i].extend(weights[1][i * nbHidden + j])
            out[i].append(weights[2][i * nbHidden + j])
            # np.concatenate((out[i], weights[1][i + j]))
            # np.append(out[i], weights[2][i + j])
    return out


def getLinearWeights(nn, nbOutput):
    weights = list(nn.parameters())
    out = []
    for i in range(2):
        weights[i] = weights[i].detach().numpy()
    for i in range(nbOutput):
        out.extend(weights[0][i])
        out.append(weights[1][i])
    return out


def main():
    df = pd.read_csv("airline.csv")
    timeseries = df[["Passengers"]].values.astype("float32")

    # plt.plot(timeseries)
    # plt.show()

    # train-test split for time series
    train_size = int(len(timeseries) * 0.67)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]

    lookback = 1
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    model = AirModel2(1, 4)
    # model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(
        data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8
    )

    n_epochs = 300
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = torch.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = torch.sqrt(loss_fn(y_pred, y_test))
        print(
            "Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse)
        )

    w1 = getLSTMWeights(model.lstm, 1, 4)
    w2 = getLinearWeights(model.linear1, 2)
    w3 = getLinearWeights(model.linear2, 1)
    np.savetxt("test1.out", w1)
    np.savetxt("test2.out", w2)
    np.savetxt("test3.out", w3)

    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(X_train)[:, -1, :]
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[train_size + lookback : len(timeseries)] = model(X_test)[:, -1, :]
    # plot
    plt.plot(timeseries, c="b")
    plt.plot(train_plot, c="r")
    plt.plot(test_plot, c="g")
    plt.show()


if __name__ == "__main__":
    main()

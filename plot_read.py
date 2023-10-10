from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"


def get_data(data_path, ext):
    files_list = []
    files = os.listdir(data_path)
    for file in files:
        if file.endswith(ext):
            files_list.append(file)

    return files_list


def split_data(data_list):
    train = []
    valid = []
    test = []
    saved_list = []
    valid_set = [37, 33, 21, 19, 17, 15, 7, 5, 2, 1]
    test_set = [39, 36, 35, 34, 32, 27, 25, 24, 18, 11]

    for file in data_list:
        saved_list.append(file)
    for i in range(len(test_set)):
        test.append(data_list.pop(test_set[i] - 1))
        valid.append(saved_list.pop(valid_set[i] - 1))
    train = [x for x in data_list if x in saved_list]

    return train, valid, test


def read_data(data_path, file_list, cols, nin, nout):
    datax = []
    datay = []
    for file in file_list:
        df = pd.read_csv(data_path + file, sep="\s+", names=cols)
        df = df.set_index("time")
        x = df.drop(columns=nout[1:5])
        y = df.drop(columns=nin[1:5])
        datax.append(x)
        datay.append(y)

    return datax, datay


def plot_data(sequences, dtype, var, folder, path):
    for i, seq in enumerate(sequences):
        seq.plot(kind="line")
        plt.legend(loc="upper left")
        Path(path + folder + dtype).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + folder + dtype + var + str(i) + "_plots.svg")
        plt.close()

def main():
    model = tf.keras.models.load_model("airline.keras")

    model.summary()

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

    saveTofile(model.layers, "airline.wei")


if __name__ == "__main__":
    main()

import kNN
import OVR
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def splitTrainTest(data, testPersent):
    trainData = []
    testData = []
    for row in data:
        if random.random() < testPersent:
            testData.append(row)
        else:
            trainData.append(row)
    return trainData, testData


def loadData(fileName):
    data = []
    # data format:
    # | pair(x: cordinate, y: cordinate) | classType |
    f = open(fileName, "r")
    for line in f:
        argList = line.split()
        data.append([[float(argList[0]), float(argList[1])], int(argList[2])])
    return data


def saveData(data, fileName):
    # data format:
    # | pair(x: cordinate, y: cordinate) | classType |
    f = open(fileName, "w")
    for i in range(len(data)):
        str = f"{data[i][0][0]}\t{data[i][0][1]}\t{data[i][1]}\n"
        f.write(str)
    f.close()


def generateData(n_Classes, n_Elem):
    # data format:
    # | pair(x: cordinate, y: cordinate) | classType |
    data = []
    for i in range(n_Classes):
        centX, centY = random.random() * 7.0, random.random() * 7.0
        for _ in range(n_Elem):
            data.append([[random.gauss(centX, 0.3), random.gauss(centY, 0.3)], i])
    return data


def randColor():
    color = np.random.rand(3)
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
    )
    return hex_color


def main():
    # loading | generating data
    classCount = 5
    elemCount = 100
    data = generateData(classCount, elemCount)
    saveData(data, f"data{classCount}Class.txt")
    print("OK")

    # data = loadData(f"data{classCount}Class.txt")
    # fingind max/min X/Y coorinate for plots
    X = np.array([i[0][0] for i in data])
    Y = np.array([i[0][1] for i in data])
    fetures = np.array([i[1] for i in data])
    xMax = max(X)
    xMin = min(X)
    yMax = max(Y)
    yMin = min(Y)
    nUniqueClasses = len(np.unique(fetures))
    colors = [randColor() for _ in range(nUniqueClasses)]
    classColors = ListedColormap(colors)
    plt.scatter(
        [data[i][0][0] for i in range(len(data))],
        [data[i][0][1] for i in range(len(data))],
        c=[data[i][1] for i in range(len(data))],
        cmap=classColors,
    )
    trainData, testData = splitTrainTest(data, 0.9)
    ovrClassifier = OVR.CustomOVRC("LogReg", nUniqueClasses)
    # split data to X(params) and y(features)
    X = np.array([i[0] for i in data])  # adding pair(x, y)
    fetures = np.array([i[1] for i in data])  # adding class type
    ovrClassifier.fit(X, fetures)

    for i in range(nUniqueClasses):
        w1, w2 = ovrClassifier.classifiers[i].weights
        b = ovrClassifier.classifiers[i].bias
        x_axis = np.linspace(xMin, xMax, len(data))
        y_axis = -(w1 * x_axis + b) / w2

        for indY, y in enumerate(y_axis):
            if y <= yMin or y >= yMax:
                np.delete(x_axis, indY)
                np.delete(y_axis, indY)

        plt.plot(
            x_axis,
            y_axis,
            color=colors[i],
            label=f"Decision Boundary {i}",
            linestyle="dashed",
        )

    plt.show()
    # knn = kNN.CustomKNN(data, 3, 150, 4, 0.6)
    # knn.calculateAccuracy()
    # knn.showDataOnMesh()


if __name__ == "__main__":
    main()

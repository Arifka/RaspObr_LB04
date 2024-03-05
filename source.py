import cv2 as cv
from matplotlib.colors import ListedColormap
import numpy as np
import random
import matplotlib.pyplot as plt
import math


def loadData(fileName):
    data = []
    f = open(fileName, "r")
    for line in f:
        argList = line.split()
        data.append([[float(argList[0]), float(argList[1])], int(argList[2])])
    return data


def splitTrainTest(data, testPercent):
    trainData = []
    testData = []
    for row in data:
        if random.random() < testPercent:
            testData.append(row)
        else:
            trainData.append(row)
    return trainData, testData


def saveData(data, fileName):
    f = open(fileName, "w")
    for i in range(len(data)):
        str = f"{data[i][0][0]}\t{data[i][0][1]}\t{data[i][1]}\n"
        f.write(str)


def generateData():
    data = []
    for i in range(3):
        centX, centY = random.random() * 5.0, random.random() * 5.0
        for row in range(150):
            data.append([[random.gauss(centX, 0.5), random.gauss(centY, 0.5)], i])
    return data


def k_neighbor(train, test, k, nClasses):
    def euqlid_norm(x, y):
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    testLabels = []
    for testPoint in test:
        # Claculate distances between test point and all of the train points
        testDist = [
            [euqlid_norm(testPoint, train[i][0]), train[i][1]]
            for i in range(len(train))
        ]
        # How many points of each class among nearest K
        stat = [0 for i in range(nClasses)]
        for d in sorted(testDist)[0:k]:
            stat[d[1]] += 1
        # Assign a class with the most number of occurences among K nearest neighbours
        testLabels.append(sorted(zip(stat, range(nClasses)), reverse=True)[0][1])
    return testLabels

def calculateAccuracy (nClasses, nItemsInClass, k, testPercent):
    data = generateData()
    trainData, testDataWithLabels = splitTrainTest (data, testPercent)
    testData = [testDataWithLabels[i][0] for i in range(len(testDataWithLabels))]
    testDataLabels = k_neighbor(trainData, testData, k, nClasses)
    print("Accuracy: ", sum([int(testDataLabels[i]==testDataWithLabels[i][1]) for i in range(len(testDataWithLabels))]) / float(len(testDataWithLabels)))


def showDataOnMesh(nClasses, nItemsInClass, k):
    # Generate a mesh of nodes that covers all train cases
    def generateTestMesh(trainData):
        x_min = min([trainData[i][0][0] for i in range(len(trainData))]) - 1.0
        x_max = max([trainData[i][0][0] for i in range(len(trainData))]) + 1.0
        y_min = min([trainData[i][0][1] for i in range(len(trainData))]) - 1.0
        y_max = max([trainData[i][0][1] for i in range(len(trainData))]) + 1.0
        h = 0.05
        testX, testY = np.meshgrid(
            np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        )
        return [testX, testY]

    trainData = generateData()
    testMesh = generateTestMesh(trainData)
    testMeshLabels = k_neighbor(
        trainData, zip(testMesh[0].ravel(), testMesh[1].ravel()), k, nClasses
    )
    classColormap = ListedColormap(["#FF0000", "#00FF00", "#FFFFFF"])
    testColormap = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAAA"])
    plt.pcolormesh(
        testMesh[0],
        testMesh[1],
        np.asarray(testMeshLabels).reshape(testMesh[0].shape),
        cmap=testColormap,
    )
    plt.scatter(
        [trainData[i][0][0] for i in range(len(trainData))],
        [trainData[i][0][1] for i in range(len(trainData))],
        c=[trainData[i][1] for i in range(len(trainData))],
        cmap=classColormap,
    )
    plt.show()


def main():
    
    calculateAccuracy(3, 150, 2, 50)

    showDataOnMesh(3, 150, 2)

if __name__ == "__main__":
    main()

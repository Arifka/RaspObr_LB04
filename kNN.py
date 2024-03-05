import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class CustomKNN:

    def __init__(self, data, nClasses, nItemsInClass, k, testPersent):
        self._data = data
        self._nClasses = nClasses
        self._nItemInClass = nItemsInClass
        self._k = k
        self._testPersent = testPersent
        self._accuracy = None

    def splitTrainTest(self):
        trainData = []
        testData = []
        for row in self._data:
            if random.random() < self._testPersent:
                testData.append(row)
            else:
                trainData.append(row)
        return trainData, testData

    def calculateAccuracy(self):
        trainData, testDataWithLabels = self.splitTrainTest()
        testData = [testDataWithLabels[i][0] for i in range(len(testDataWithLabels))]
        testLabels = self.k_neighbor(trainData, testData)
        self._accuracy = sum(
            [
                int(testLabels[i] == testDataWithLabels[i][1])
                for i in range(len(testDataWithLabels))
            ]
        ) / float(len(testDataWithLabels))
        print("Accuracy: ", self._accuracy)

    def k_neighbor(self, trainData, testData):
        def euqlid_norm(x, y):
            return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

        testLabels = []

        for testPoint in testData:
            # Claculate distances between test point and all of the train points
            testDist = [
                [euqlid_norm(testPoint, trainData[i][0]), trainData[i][1]]
                for i in range(len(trainData))
            ]
            # testDist format:
            # | euqlid norm between testPoint and labled point | class of labled point |
            # How many points of each class among nearest K
            stat = [0 for i in range(self._nClasses)]
            for d in sorted(testDist)[0 : self._k]:
                stat[d[1]] += 1
            # Assign a class with the most number of occurences among K nearest neighbours
            testLabels.append(
                sorted(zip(stat, range(self._nClasses)), reverse=True)[0][1]
            )
        return testLabels

    def showDataOnMesh(self):
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

        testMesh = generateTestMesh(self._data)
        testMeshLabels = self.k_neighbor(
            self._data, zip(testMesh[0].ravel(), testMesh[1].ravel())
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
            [self._data[i][0][0] for i in range(len(self._data))],
            [self._data[i][0][1] for i in range(len(self._data))],
            c=[self._data[i][1] for i in range(len(self._data))],
            cmap=classColormap,
        )
        plt.show()

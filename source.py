import cv2 as cv
import kNN
import numpy as np
import random
import matplotlib.pyplot as plt
import math


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


def generateData():
    # data format:
    # | pair(x: cordinate, y: cordinate) | classType |
    data = []
    for i in range(3):
        centX, centY = random.random() * 5.0, random.random() * 5.0
        for row in range(150):
            data.append([[random.gauss(centX, 0.5), random.gauss(centY, 0.5)], i])
    return data


def main():

    data = loadData("data.txt")
    knn = kNN.CustomKNN(data, 3, 150, 4, 0.6)
    knn.calculateAccuracy()
    knn.showDataOnMesh()


if __name__ == "__main__":
    main()

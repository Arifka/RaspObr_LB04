import cv2 as cv
from matplotlib.colors import ListedColormap
import numpy as np
import random
import matplotlib.pyplot as plt
import math

def splitData(data, testPercent):
    trainData = []
    testData = []
    pass


def k_neighbor():
    def euqlid_norm(x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    pass


def main():
    data = []
    for i in range(3):
        centX, centY = random.random()*3.0, random.random()*3.0
        for row in range(150):
            data.append([[random.gauss(centX, 0.5), random.gauss(centY, 0.5)], i])

    colorMap = ListedColormap(['#ff0000', '#00ff00', '#ff00ff'])

    data1_x = 3 + np.random.normal(1, 0.7, 150)
    data1_y = 1 - np.random.normal(0, 1, len(data1_x))

    data2_x = np.random.normal(0, 0.7, 150)
    data2_y = 1 + np.random.normal(0, 1, len(data2_x))

    data3_x = 2 - np.random.normal(0, 0.5, 150)
    data3_y = 4 + np.random.normal(0, 1, len(data3_x))

    # plt.subplot(211)
    # plt.scatter(data1_x, data1_y, c="green")
    # plt.scatter(data2_x, data2_y, c="red")
    # plt.scatter(data3_x, data3_y, c="blue")

    #plt.subplot(221)
    plt.scatter([data[i][0][0] for i in range(len(data))], 
                [data[i][0][1] for i in range(len(data))], 
                c=[data[i][1] for i in range(len(data))], 
                cmap=colorMap)

    plt.show()
    

if __name__ == '__main__':
    main()
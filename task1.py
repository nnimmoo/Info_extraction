import cv2
import numpy as np


def extractAllInfo(image, threshold):
    image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
    image = np.flip(image, axis=1)
    x_value, y_value = image.shape
    points_list = []
    for x in range(x_value - 1):
        for y in range(y_value - 1):
            if image[x, y] < threshold:
                points_list.append((-x, -y))
    return points_list


def filterInfo(points):
    temp = []
    lst = list(np.unique([item[1] for item in points]))
    for x in points:
        if lst.count(x[1]) != 0:
            temp.append(x)
            lst.remove(x[1])
    return temp


def construct(image, points):
    X, Y, _ = image.shape
    image = np.zeros((X, Y, 3), dtype=np.uint8)
    for x in points:
        image[x[0], x[1], :] = (0xFF, 0xFF, 0xFF)
    image[200, 100, :] = (0x0, 0x0, 0xFF)
    cv2.imwrite('output.png', image)

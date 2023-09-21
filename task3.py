import pylab as plt
from task1 import *


def leastSquare(degree, y_points, x_points):
    x = np.array([])
    y = np.array([])

    for j in range(0, (degree + 1)):
        for k in range(0, (degree + 1)):
            x_temp = np.sum(x_points ** (j + k))
            x = np.append(x, x_temp)
        y_temp = np.sum(y_points * (x_points ** j))
        y = np.append(y, y_temp)

    x = np.reshape(x, ((degree + 1), (degree + 1)))
    y = np.reshape(y, ((degree + 1), 1))
    a = np.linalg.solve(x, y)
    kin = 0

    for i in range(0, np.size(a)):
        kin += (a[i] * (x_points ** i))

    plt.plot(x_points, y_points, '-', label="function")
    plt.plot(x_points, kin, color='red', label='lst sq')
    plt.legend(loc='upper left')
    plt.show()


points = filterInfo(extractAllInfo("test.png", 150))
points = (sorted(points, key=lambda point: point[1]))
y = np.array([item[0] for item in points])
x = np.array([item[1] for item in points])
leastSquare(1, y, x)

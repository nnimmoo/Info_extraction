import pylab
from task1 import *


def peakDetection(y, l, threshold, inf=0):
    peak = np.zeros(len(y))
    newY = np.array(y)
    avg = [0] * len(y)
    stdev = [0] * len(y)
    avg[l - 1] = np.mean(y[0:l])
    stdev[l - 1] = np.std(y[0:l])
    for i in range(l, len(y) - 1):
        if abs(y[i] - avg[i - 1]) < threshold * stdev[i - 1]:
            peak[i] = 0
            newY[i] = y[i]
            avg[i] = np.mean(newY[(i - l):i])
            stdev[i] = np.std(newY[(i - l):i])
        else:
            if y[i] > avg[i - 1]:
                peak[i] = 1
            else:
                peak[i] = 0
            newY[i] = inf * y[i] + (1 - inf) * newY[i - 1]
            avg[i] = np.mean(newY[(i - l):i])
            stdev[i] = np.std(newY[(i - l):i])
    return np.asarray(peak)

# tester
points = filterInfo(extractAllInfo("test.png", 150))

points = (sorted(points, key=lambda point: point[1]))
y = np.array([item[0] for item in points])
x = np.array([item[1] for item in points])

result = peakDetection(y, 30, 4.63)

pylab.subplot(211)
pylab.plot(np.arange(1, len(y) + 1), y)
pylab.subplot(212)
pylab.step(np.arange(1, len(y) + 1), result, color="red", lw=2)
pylab.show()

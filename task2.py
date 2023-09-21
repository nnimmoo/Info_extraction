import numpy as np


def extractMatrix(matrix):
    lowerDiagonal,middleDiagonal,upperDiagonal = [],[],[]
    for x in range(len(matrix)):
        middleDiagonal.append(matrix[x][x]);
    for x in range(len(matrix) - 1):
        upperDiagonal.append(matrix[x][x + 1]);
    for x in range(len(matrix) - 1):
        lowerDiagonal.append(matrix[x + 1][x]);
    return lowerDiagonal, middleDiagonal, upperDiagonal


def thomasAlgo(matrix, vector):
    lowerDiagonal, middleDiagonal, upperDiagonal = extractMatrix(matrix)
    a, b, c, d = map(np.array, (lowerDiagonal, middleDiagonal, upperDiagonal, vector))
    for i in range(1, len(vector)):
        mc = a[i - 1] / b[i - 1]
        b[i] = b[i] - mc * c[i - 1]
        d[i] = d[i] - mc * d[i - 1]
    ret = b
    ret[-1] = d[-1] / b[-1]
    for j in range(len(vector) - 2, -1, -1):
        ret[j] = (d[j] - c[j] * ret[j + 1]) / b[j]
    return ret


def jacobi(A, b, x0, t, iter=100):
    n, x, x1, counter, xd = A.shape[0], x0.copy(), x0.copy(), 0, t + 1
    while (xd > t) and (counter < iter):
        for i in range(0, n):
            s = 0
            for j in range(0, n):
                if i != j:
                    s += A[i, j] * x1[j]
            x[i] = (b[i] - s) / A[i, i]
        counter += 1
        xd = (np.sum((x - x1) * 2)) * 0.5
        x1 = x.copy()
    return x

def cubicSpline(x, y, tol=1e-100):
    x, y = np.array(x), np.array(y)

    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

    dim, x_temp, y_temp = len(x), np.diff(x), np.diff(y)

    a, b = np.zeros(shape=(dim, dim)), np.zeros(shape=(dim, 1))
    a[0, 0] = 1
    a[-1, -1] = 1

    for i in range(1, dim - 1):
        a[i, i + 1] = x_temp[i]
        a[i, i - 1] = x_temp[i - 1]
        a[i, i] = 2 * (x_temp[i - 1] + x_temp[i])
        b[i, 0] = 3 * (y_temp[i] / x_temp[i] - y_temp[i - 1] / x_temp[i - 1])

    c = jacobi(a, b, np.zeros(len(a)), tol, 1500)
    d, b = np.zeros(shape=(dim - 1, 1)), np.zeros(shape=(dim - 1, 1))
    for i in range(0, len(d)):
        d[i] = (c[i + 1] - c[i]) / (3 * x_temp[i])
        b[i] = (y_temp[i] / x_temp[i]) - (x_temp[i] / 3) * (2 * c[i] + c[i + 1])

    return b.squeeze(), c.squeeze(), d.squeeze()

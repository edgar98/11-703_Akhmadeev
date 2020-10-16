import matplotlib.pyplot as plt
import numpy as np


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


n = 10000
eps, minPts = 5, 3
flags = []
x = np.random.randint(1, 100, n)
y = np.random.randint(1, 100, n)
for i in range(0,n):
    neighb = -1
    for j in range(0, n):
        if distance(x[i], y[i], x[j], y[j]) <= eps:
            neighb += 1
    if neighb >= minPts:
        flags.append('g')
    else:
        flags.append('r')
for i in range(0, n):
    if flags[i] != 'g':
        for j in range(0, n):
            if flags[j] == 'g':
                if distance(x[i], y[i], x[j], y[j]) < eps:
                    flags[i] = 'y'
                    break
for i in range(0,n):
    plt.scatter(x[i], y[i], color=flags[i])

plt.show()
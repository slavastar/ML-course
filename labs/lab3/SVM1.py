import numpy as np
from matplotlib import pyplot as plt


def print_hyperplane(w):
    x2 = [w[0], w[1], -w[1], w[0]]
    x3 = [w[0], w[1], w[1], -w[0]]
    x2x3 = np.array([x2, x3])
    X, Y, U, V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale=1, color='blue')
    plt.show()


def svm_sgd_plot(X, Y):
    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 100000
    for epoch in range(1, epochs):
        for i, x in enumerate(X):
            if (Y[i] * np.dot(X[i], w)) < 1:
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1 / epoch) * w))
            else:
                w = w + eta * (-2 * (1 / epoch) * w)
    return w


X = np.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
])

y = np.array([-1, -1, 1, 1, 1])

for d, sample in enumerate(X):
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

w = svm_sgd_plot(X, y)
print("w =", w)
print_hyperplane(w)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kernel_linear(a, b):
    return np.dot(a, b)


def kernel_polynomial(a, b, gamma, c, degree):
    return np.power(c + gamma * np.dot(a, b), degree)


def kernel_gaussian(a, b, gamma=0.5):
    return np.exp(-gamma * np.power(np.linalg.norm(a - b), 2))


def read_dataset(filename):
    data = pd.read_csv(filename)
    X = data.values[:, :-1]
    Y = np.vectorize(lambda t: 1 if t == 'P' else -1)(data.values[:, -1])
    indices = np.arange(len(Y))
    np.random.shuffle(indices)
    return X[indices], Y[indices]


X, Y = read_dataset("datasets/chips.csv")
plt.scatter(X[:, 0], X[:, 1])
plt.show()

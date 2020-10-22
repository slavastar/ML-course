import numpy as np
import random
from matplotlib import pyplot as plt


def scalar_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def SMAPE(X, Y, w):
    result = 0
    for i in range(len(X)):
        y_predict = scalar_product(X[i], w)
        y_real = Y[i]
        # print("predict =", y_predict, "\treal = ", y_real)
        result += abs(y_predict - y_real) / (abs(y_predict) + abs(y_real))
    return result * 200 / len(X)


def LSM(X, Y, tau):
    return np.linalg.pinv(np.add(X.T.dot(X), np.cov(X.T).dot(tau))).dot(X.T).dot(Y)


def find_minmax_X(X):
    minmax = np.zeros((len(X[0]), 2))
    for i in range(len(X[0])):
        column = X[:, i]
        minmax[i] = [column.min(), column.max()]
    return minmax


def find_minmax_Y(Y):
    return np.array([Y.min(), Y.max()])


def normalize(X, Y, with_constant_feature=True):
    X_normalized, Y_normalized = np.zeros(np.shape(X)), np.zeros(np.shape(Y))
    minmax_X, minmax_Y = find_minmax_X(X), find_minmax_Y(Y)
    for i in range(len(X)):
        Y_normalized[i] = (Y[i] - minmax_Y[0]) / (minmax_Y[1] - minmax_Y[0])
        for j in range(len(X[0])):
            if with_constant_feature and j == 0:
                X_normalized[i][0] = 1
            elif minmax_X[j][0] == minmax_X[j][1]:
                X_normalized[i][j] = 0
            else:
                X_normalized[i][j] = (X[i][j] - minmax_X[j][0]) / (minmax_X[j][1] - minmax_X[j][0])
    return X_normalized, Y_normalized


def denormalize_weights(w_normalized, X, Y):
    w = np.zeros(np.shape(w_normalized))
    minmax_X, minmax_Y = find_minmax_X(X), find_minmax_Y(Y)
    y_scale = np.amax(minmax_Y) - np.amin(minmax_Y)
    w[0] = minmax_Y[0]
    for i in range(len(X[0])):
        if minmax_X[i][1] > minmax_X[i][0]:
            w[0] -= minmax_X[i][0] * w_normalized[i] * y_scale / (minmax_X[i][1] - minmax_X[i][0])
    for i in range(1, len(X[0])):
        if minmax_X[i][1] > minmax_X[i][0]:
            w[i] = w_normalized[i] * y_scale / (minmax_X[i][1] - minmax_X[i][0])
    w[0] = minmax_Y[0]
    return w


def MSE(predict, real):
    return (predict - real) ** 2


def MSE_derivative(predict, real):
    return 2 * (predict - real)


def initialise_Q(X, Y, w):
    Q = 0
    for i in range(len(X)):
        Q += MSE(scalar_product(X[i], w), Y[i])
    return Q / len(X)


def initialise_weights(features, objects):
    w = np.zeros(features)
    for i in range(features):
        w[i] = random.uniform(-0.5 / objects, 0.5 / objects)
    return w


def SGD(X, Y, tau, iterations_limit=1000, with_Q_limit=True):
    objects, features = len(X), len(X[0])
    w = initialise_weights(features, objects)
    Q = initialise_Q(X, Y, w)
    iterations = 0
    while True:
        iterations += 1
        k = random.randint(0, objects - 1)
        x, y = X[k], Y[k]
        predict = scalar_product(x, w)
        loss_value = MSE(predict, y) + 0.5 * tau * np.linalg.norm(w)
        learning_rate = 0.007 / iterations
        for i in range(features):
            gradient = MSE_derivative(predict, y) * x[i]
            w[i] = w[i] * (1 - learning_rate * tau) - learning_rate * gradient
        decay = 0.035
        Q_previous = Q
        Q = (1 - decay) * Q + decay * loss_value
        if iterations == iterations_limit or (abs(Q - Q_previous) < 0.000001 and with_Q_limit):
            print("tau = ", tau, "\tnumber of iterations =", iterations)
            break
    return w


def add_constant_feature(X, feature_value=1):
    X_new = np.zeros((len(X), len(X[0]) + 1))
    for i in range(len(X)):
        X_new[i][0] = feature_value
        for j in range(len(X[0])):
            X_new[i][j + 1] = X[i][j]
    return X_new


def process_file(filename):
    with open(filename) as file:
        features = int(next(file))
        objects_train = int(next(file))
        X_train = np.zeros((objects_train, features))
        Y_train = np.zeros(objects_train)
        for i in range(objects_train):
            line = [int(x) for x in next(file).split()]
            Y_train[i] = line.pop()
            X_train[i] = line
        objects_test = int(next(file))
        X_test = np.zeros((objects_test, features))
        Y_test = np.zeros(objects_test)
        for i in range(objects_test):
            line = [int(x) for x in next(file).split()]
            Y_test[i] = line.pop()
            X_test[i] = line
        return X_train, Y_train, X_test, Y_test


def find_optimal_tau(X, Y, algorithm):
    tau_values = [10 ** -i for i in range(0, 10)]
    optimal_tau, optimal_w, min_smape = 0, [], 200
    for tau in tau_values:
        w = algorithm(X, Y, tau)
        smape = SMAPE(X, Y, w)
        if smape < min_smape:
            optimal_tau, optimal_w, min_smape = tau, w, smape
    return optimal_tau, optimal_w, min_smape


def get_optimal_tau(X_train, Y_train, X_test, Y_test, algorithm):
    optimal_tau_train, optimal_w_train, min_smape_train = find_optimal_tau(X_train, Y_train, algorithm)
    print("optimal tau =", optimal_tau_train, "\tmin smape train =", min_smape_train)
    print("smape test =", SMAPE(X_test, Y_test, optimal_w_train))
    return optimal_tau_train


def process_dataset_for_SGD(X_train, Y_train, X_test, Y_test):
    add_constant_feature(X_train)
    add_constant_feature(X_test)
    X_train, Y_train = normalize(X_train, Y_train)
    X_test, Y_test = normalize(X_test, Y_test)
    return X_train, Y_train, X_test, Y_test


def draw_graph(x_values, y_values, x_name, y_name):
    plt.plot(x_values, y_values, 'b')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


algorithm = SGD
X_train, Y_train, X_test, Y_test = process_file("datasets/1.txt")
if algorithm == SGD:
    X_train, Y_train, X_test, Y_test = process_dataset_for_SGD(X_train, Y_train, X_test, Y_test)
optimal_tau = get_optimal_tau(X_train, Y_train, X_test, Y_test, algorithm)
print(optimal_tau)
if algorithm == SGD:
    iterations_limit_values = [100 * i for i in range(1, 21)]
    smape_values = []
    for iteration_limit_value in iterations_limit_values:
        w = SGD(X_train, Y_train, optimal_tau, iteration_limit_value, False)
        smape_values.append(SMAPE(X_test, Y_test, w))
    draw_graph(iterations_limit_values, smape_values, "iterations", "smape")

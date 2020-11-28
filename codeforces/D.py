from math import e, log2, log
import random
import numpy as np


def print_answer(w, b):
    for element in w:
        print(element)
    print(b)


def SMAPE(X, Y, w, b):
    result = 0
    for i in range(objects):
        y_predict = b + scalar_product(w, X[i])
        y_real = Y[i]
        result += abs(y_predict - y_real) / (abs(y_predict) + abs(y_real))
    return result * 200 / objects


def print_X(X):
    for row in X:
        print(row)


def minmax_X(X):
    minmax = list()
    for i in range(len(X[0])):
        min_value = max_value = X[0][i]
        for j in range(1, len(X)):
            if X[j][i] > max_value:
                max_value = X[j][i]
            if X[j][i] < min_value:
                min_value = X[j][i]
        minmax.append([min_value, max_value])
    return minmax


def normalize_X(X, minmax):
    normalized_X = []
    for j in range(len(X)):
        row = []
        for i in range(len(X[0])):
            if minmax[i][0] == minmax[i][1]:
                row.append(0)
            else:
                row.append(float((X[j][i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])))
        normalized_X.append(list(row))
    return normalized_X


def minmax_Y(Y):
    return min(Y), max(Y)


def normalize_Y(Y):
    min_value, max_value = minmax_Y(Y)
    normalized_Y = []
    for i in range(len(Y)):
        normalized_Y.append(float((Y[i] - min_value) / (max_value - min_value)))
    return normalized_Y


def denormalize_weights(normalized_w, normalized_b, minmax_X, minmax_Y):
    w = []
    x_scale = []
    for i in range(len(minmax_X)):
        x_scale.append(minmax_X[i][1] - minmax_X[i][0])
    y_scale = minmax_Y[1] - minmax_Y[0]
    for i in range(features):
        if x_scale[i] == 0:
            w.append(0)
        else:
            w.append(normalized_w[i] * y_scale / x_scale[i])
    b = normalized_b + minmax_Y[0]
    for i in range(features):
        if x_scale[i] != 0:
            b -= minmax_X[i][0] * y_scale / x_scale[i] * normalized_w[i]
    return w, b


def initialise_weights():
    b = random.uniform(-0.5 / objects, 0.5 / objects)
    w = []
    for i in range(features):
        w.append(random.uniform(-0.5 / objects, 0.5 / objects))
    return w, b


def are_vectors_similar(a, b):
    for i in range(len(a)):
        if abs(a[i] - b[i]) > 0.000001:
            return False
    return True


def scalar_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def MSE(y_predict, y_real):
    return (y_predict - y_real) ** 2


def MSE_derivative(y_predict, y_real):
    return 2 * (y_predict - y_real)


def f(y_predict, y_real):
    return abs(y_predict - y_real)


def f_derivative(y_predict, y_real):
    if y_predict > y_real:
        return 1
    elif y_predict < y_real:
        return -1
    else:
        return 0


def vector_length(w, b):
    length = 0
    for component in w:
        length += component ** 2
    length += b ** 2
    return length ** 0.5


def vector_length_squared(w, b):
    return vector_length(w, b) ** 2


loss_functions = {
    1: [MSE, MSE_derivative],
    2: [f, f_derivative]
}


def SGD(X, Y, loss_function, tau=0, max_number_of_iterations=1000):
    # parameters
    alpha = 0.05
    batch_size = min(1, objects)

    # initialisation
    w, b = initialise_weights()
    Q = 0
    for i in range(objects):
        Q += loss_function[0](scalar_product(w, X[i]), Y[i])
    Q /= objects

    # finding max value in X
    minmax = minmax_X(X)
    max_value = 0
    for minmax_element in minmax:
        if minmax_element[1] > max_value:
            max_value = minmax_element[1]

    iterations = 0
    while iterations < max_number_of_iterations:
        iterations += 1
        learning_rate = 0.5 / iterations
        indices = random.sample(range(0, objects), batch_size)
        total_w = [0] * features
        total_b = 0
        total_loss_value = 0

        for index in indices:
            x, y_real = X[index], Y[index]
            y_predict = b + scalar_product(w, x)
            total_loss_value += loss_function[0](y_predict, y_real)
            loss_derivative_value = loss_function[1](y_predict, y_real)
            total_b += loss_derivative_value
            for i in range(features):
                total_w[i] += loss_derivative_value * x[i]

        loss_value = total_loss_value / batch_size + 0.5 * tau * vector_length_squared(w, b)
        b = b * (1 - learning_rate * tau) - learning_rate * total_b / batch_size
        w_previous = w.copy()
        for i in range(features):
            w[i] = w[i] * (1 - learning_rate * tau) - learning_rate * total_w[i] / batch_size

        # print("iteration", iterations, "\tloss value =", round(loss_value, 4), "b =", round(b, 4), "\tw =", w)

        Q_previous = Q
        Q = (1 - alpha) * Q + alpha * loss_value
        if are_vectors_similar(w_previous, w) or abs(Q - Q_previous) < 0.000001:
            break
    return w, b


def LSM(X, Y, tau):
    return np.linalg.pinv(np.add(X.T.dot(X), np.cov(X.T).dot(tau))).dot(X.T).dot(Y)


objects, features = map(int, input().split())
X = np.zeros((objects, features + 1))
Y = np.zeros(objects)
for i in range(objects):
    line = list(map(int, input().split()))
    # Y.append(line.pop(len(line) - 1))
    # X.append(line)
    Y[i] = line.pop()
    line.append(1)
    X[i] = line
w = LSM(X, Y, 1)
print_answer(w[0:len(w) - 1], w[len(w) - 1])
# normalized_X = normalize_X(X, minmax_X(X))
# normalized_Y = normalize_Y(Y)
# print("X")
# print_X(X)
# print("Y")
# print(Y)
# print("Normalized X")
# print_X(normalized_X)
# print("Normalized Y")
# print(normalized_Y)
# normalized_w, normalized_b = SGD(normalized_X, normalized_Y, loss_functions[1])
# w, b = denormalize_weights(normalized_w, normalized_b, minmax_X(X), minmax_Y(Y))
# print("Result")
# print("b =", b, "\tw =", w)
# print_answer(w, b)

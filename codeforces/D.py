from math import e, log2, log
import random


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
            elif X[j][i] < min_value:
                min_value = X[j][i]
        minmax.append([min_value, max_value])
    return minmax


def normalize_X(X, minmax):
    normalized_X = []
    for j in range(len(X)):
        row = []
        for i in range(len(X[0])):
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


def SGD(X, Y):
    w, b = initialise_weights()

    # initialise Q
    Q = 0
    for i in range(objects):
        Q += MSE(scalar_product(w, X[i]), Y[i])
    Q /= objects

    alpha = 0.05
    iterations = 0
    batch_size = min(1, objects)
    while True:
        iterations += 1
        learning_rate = 0.003
        indices = random.sample(range(0, objects), batch_size)
        total_w = [0] * features
        total_b = 0
        total_loss_value = 0
        for index in indices:
            x, y_real = X[index], Y[index]
            y_predict = b + scalar_product(w, x)
            total_loss_value += MSE(y_predict, y_real)
            # update weights
            MSE_derivative_value = MSE_derivative(y_predict, y_real)
            total_b += learning_rate * MSE_derivative_value
            for i in range(features):
                total_w[i] += learning_rate * MSE_derivative_value * x[i]

        loss_value = total_loss_value / batch_size
        b -= total_b / batch_size
        w_previous = w.copy()
        for i in range(features):
            w[i] -= total_w[i] / batch_size
        # print("iteration", iterations, "\tloss value =", round(loss_value, 4), "b =", round(b, 4), "\tw =", w)

        Q_previous = Q
        Q = (1 - alpha) * Q + alpha * loss_value
        if iterations == 2000 or (iterations > 10 and are_vectors_similar(w_previous, w) or abs(Q - Q_previous) < 0.00001):
            break
    return w, b


objects, features = map(int, input().split())
X = []
Y = []
for i in range(objects):
    line = list(map(int, input().split()))
    Y.append(line.pop(len(line) - 1))
    X.append(line)
normalized_X = normalize_X(X, minmax_X(X))
normalized_Y = normalize_Y(Y)
# print("X")
# print_X(X)
# print("Y")
# print(Y)
# print("Normalized X")
# print_X(normalized_X)
# print("Normalized Y")
# print(normalized_Y)
w, b = SGD(X, Y)
for component in w:
    print(component)
print(b)
# w, b = SGD(normalized_X, normalized_Y)
# print("Result")
# print("b =", b, "\tw =", w)
# print("SMAPE =", SMAPE(X, Y, w, b), "%")
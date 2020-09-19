import math

import pandas as pd
import numpy
from matplotlib import pyplot as plt


def minkowski_distance(x, y, p):
    distance = 0
    for i in range(len(x)):
        distance += abs(x[i] - y[i])**p
    return distance**(1 / p)


def manhattan_distance(x, y):
    return minkowski_distance(x, y, 1)


def euclidean_distance(x, y):
    return minkowski_distance(x, y, 2)


def chebyshev_distance(x, y):
    distance = 0
    for i in range(len(x)):
        distance = max(distance, abs(x[i] - y[i]))
    return distance


def distance(distance_function, x, y):
    if distance_function == 'manhattan':
        return manhattan_distance(x, y)
    elif distance_function == 'euclidean':
        return euclidean_distance(x, y)
    elif distance_function == 'chebyshev':
        return chebyshev_distance(x, y)


def bad_argument(u):
    return abs(u) >= 1


def uniform(u):
    if bad_argument(u):
        return 0
    return 0.5


def triangular(u):
    if bad_argument(u):
        return 0
    return 1 - abs(u)


def epanechnikov(u):
    if bad_argument(u):
        return 0
    return 3 / 4 * (1 - u**2)


def quartic(u):
    if bad_argument(u):
        return 0
    return 15 / 16 * (1 - u**2)**2


def triweight(u):
    if bad_argument(u):
        return 0
    return 35 / 32 * (1 - u**2)**3


def tricube(u):
    if bad_argument(u):
        return 0
    return 70 / 81 * (1 - abs(u)**3)**3


def gaussian(u):
    return 1 / (2 * math.pi)**0.5 * math.e**(-0.5 * u**2)


def cosine(u):
    if bad_argument(u):
        return 0
    return math.pi / 4 * math.cos(math.pi / 2 * u)


def logistic(u):
    return 1 / (math.e**u + 2 + math.e**(-u))


def sigmoid(u):
    return 2 / math.pi / (math.e**u + math.pi**(-u))


def kernel(kernel_function, u):
    if kernel_function == 'uniform':
        return uniform(u)
    elif kernel_function == 'triangular':
        return triangular(u)
    elif kernel_function == 'epanechnikov':
        return epanechnikov(u)
    elif kernel_function == 'quartic':
        return quartic(u)
    elif kernel_function == 'triweight':
        return triweight(u)
    elif kernel_function == 'tricube':
        return tricube(u)
    elif kernel_function == 'gaussian':
        return gaussian(u)
    elif kernel_function == 'cosine':
        return cosine(u)
    elif kernel_function == 'logistic':
        return logistic(u)
    elif kernel_function == 'sigmoid':
        return sigmoid(u)


def harmonic_mean(a, b):
    return divide(2 * a * b, (a + b))


def divide(a, b):
    if b == 0:
        return 0
    return a / b


def minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        if i == len(dataset[0]) - 1:
            continue
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    normalized_dataset = []
    for row in dataset:
        for i in range(len(row)):
            if i == len(row) - 1:
                continue
            row[i] = float((row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]))
        normalized_dataset.append(list(row))
    return normalized_dataset


def normalize_vector(vector, minmax):
    normalized_vector = []
    for i in range(len(vector)):
        normalized_vector.append(float((vector[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])))
    return normalized_vector


def predict(dataset, target):
    sorted_dataset = sorted(dataset, key=lambda row: distance(distance_function, row[0:len(row) - 1], target))
    numerator = 0
    denominator = 0
    for j in range(neighbors):
        kernel_value = kernel(kernel_function, j / h)
        numerator += sorted_dataset[j][features] * kernel_value
        denominator += kernel_value
    result = numerator / denominator
    return result


def get_F_score(confusion_matrix):
    C = []
    P = []
    TP = []
    FP = []
    FN = []
    precision = []
    recall = []
    F = []
    for i in range(len(confusion_matrix)):
        TP.append(confusion_matrix[i][i])
        row = confusion_matrix[i]
        column = []
        for j in range(len(confusion_matrix)):
            column.append(confusion_matrix[j][i])
        sum_row = sum(row)
        sum_column = sum(column)
        P.append(sum_row)
        C.append(sum_column)
        FP.append(sum_row - confusion_matrix[i][i])
        FN.append(sum_column - confusion_matrix[i][i])
        precision.append(divide(TP[i], (TP[i] + FP[i])))
        recall.append(divide(TP[i], (TP[i] + FN[i])))
        F.append(harmonic_mean(precision[i], recall[i]))
    return F


def regression(dataset):
    min_max = minmax(dataset)
    dataset = normalize(dataset, min_max)
    instances = len(dataset)
    true = false = 0
    confusion_matrix = [[0 for j in range(len(classes))] for i in range(len(classes))]
    for i in range(len(dataset)):
        dataset_train = dataset.copy()
        dataset_test = dataset_train.pop(i)
        prediction = round(predict(dataset_train, dataset_test[0:len(dataset_test) - 1]))
        real = dataset_test[len(dataset_test) - 1]
        confusion_matrix[number_to_index[prediction]][number_to_index[real]] += 1
        if prediction == real:
            true += 1
        else:
            false += 1
    print("----------")
    print("Instances: " + str(instances))
    print("True: " + str(true))
    print("False: " + str(false))
    print(confusion_matrix)
    print(get_F_score(confusion_matrix))


def vectorize_dataset(dataset):
    for i in range(len(dataset)):
        dataset[i][features] = mapping[dataset[i][features]]


mapping = {
    'L': 1.0,
    'B': 2.0,
    'R': 3.0
}

number_to_index = {
    1.0: 0,
    2.0: 1,
    3.0: 2
}

distance_function = 'euclidean'
kernel_function = 'cosine'
neighbors = 4
h = neighbors
classes = [1.0, 2.0, 3.0]
filename = 'data2.csv'
data = pd.read_csv(filename)
dataset = data.values
features = len(dataset[0]) - 1
vectorize_dataset(dataset)
regression(dataset)
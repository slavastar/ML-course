import math

import pandas as pd
import numpy
from matplotlib import pyplot as plt


def print_matrix(matrix):
    for row in matrix:
        print(row)


def minkowski(x, y, p):
    distance = 0
    for i in range(len(x)):
        distance += abs(x[i] - y[i])**p
    return distance**(1 / p)


def manhattan(x, y):
    return minkowski(x, y, 1)


def euclidean(x, y):
    return minkowski(x, y, 2)


def chebyshev(x, y):
    distance = 0
    for i in range(len(x)):
        distance = max(distance, abs(x[i] - y[i]))
    return distance


def check_argument(u):
    return abs(u) >= 1


def uniform(u):
    if check_argument(u):
        return 0
    return 0.5


def triangular(u):
    if check_argument(u):
        return 0
    return 1 - abs(u)


def epanechnikov(u):
    if check_argument(u):
        return 0
    return 3 / 4 * (1 - u**2)


def quartic(u):
    if check_argument(u):
        return 0
    return 15 / 16 * (1 - u**2)**2


def triweight(u):
    if check_argument(u):
        return 0
    return 35 / 32 * (1 - u**2)**3


def tricube(u):
    if check_argument(u):
        return 0
    return 70 / 81 * (1 - abs(u)**3)**3


def gaussian(u):
    return 1 / (2 * math.pi)**0.5 * math.e**(-0.5 * u**2)


def cosine(u):
    if check_argument(u):
        return 0
    return math.pi / 4 * math.cos(math.pi / 2 * u)


def logistic(u):
    return 1 / (math.e**u + 2 + math.e**(-u))


def sigmoid(u):
    return 2 / math.pi / (math.e**u + math.pi**(-u))


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


def average_value(dataset):
    result = 0
    for row in dataset:
        if isinstance(row[len(row) - 1], list):
            result += row[len(row) - 1].index(max(row[len(row) - 1])) + 1
        else:
            result += row[len(row) - 1]
    return result / instances


def predict(dataset, target, distance_function, kernel_function, window, is_variable):
    sorted_dataset = sorted(dataset, key=lambda row: distance_function(row[0:len(row) - 1], target))
    if is_variable:
        window = distance_function(sorted_dataset[window][0:len(sorted_dataset[window]) - 1], target)
    if isinstance(sorted_dataset[0][len(dataset[0]) - 1], list):
        numerator = [0 for i in range(len(sorted_dataset[0][len(dataset[0]) - 1]))]
        denominator = 0
        for j in range(len(dataset)):
            kernel_value = kernel_function(distance_function(sorted_dataset[j][0:len(sorted_dataset[j]) - 1], target) / window)
            vector = sorted_dataset[j][len(dataset[0]) - 1]
            numerator = [numerator[i] + vector[i] * kernel_value for i in range(len(numerator))]
            denominator += kernel_value
        if denominator != 0:
            result_vector = [numerator[i] / denominator for i in range(len(numerator))]
            return result_vector.index(max(result_vector)) + 1
        else:
            return average_value(sorted_dataset)
    else:
        numerator = denominator = 0
        for j in range(len(dataset)):
            kernel_value = kernel_function(distance_function(sorted_dataset[j][0:len(sorted_dataset[j]) - 1], target) / window)
            numerator += sorted_dataset[j][len(dataset[0]) - 1] * kernel_value
            denominator += kernel_value
        if denominator != 0:
            return numerator / denominator
        else:
            return average_value(sorted_dataset)


def get_F_score(confusion_matrix):
    C = []
    P = []
    TP = []
    FP = []
    FN = []
    precision = []
    recall = []
    F = []
    micro_F = 0
    precision_w = 0
    recall_w = 0
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
        micro_F += C[i] * F[i]
        precision_w += divide(TP[i] * C[i], P[i])
        recall_w += TP[i]
    micro_F /= instances
    precision_w /= instances
    recall_w /= instances
    macro_F = harmonic_mean(precision_w, recall_w)
    return macro_F


def regression(dataset, distance_function, kernel_function, window, is_variable):
    min_max = minmax(dataset)
    dataset = normalize(dataset, min_max)
    true = false = 0
    confusion_matrix = [[0 for j in range(len(classes))] for i in range(len(classes))]
    for i in range(len(dataset)):
        dataset_train = dataset.copy()
        dataset_test = dataset_train.pop(i)
        prediction = predict(dataset_train, dataset_test[0:len(dataset_test) - 1], distance_function, kernel_function, window, is_variable)
        prediction = round(prediction)
        real = dataset_test[len(dataset_test) - 1]
        if isinstance(real, list):
            real = real.index(max(real)) + 1
        confusion_matrix[number_to_index[prediction]][number_to_index[real]] += 1
        if prediction == real:
            true += 1
        else:
            false += 1
    print("----------")
    print("Distance: " + str(distance_function.__name__))
    print("Kernel: " + str(kernel_function.__name__))
    print("Window: " + str(window))
    if is_variable:
        print("Window type: variable")
    else:
        print("Window type: fixed")
    print("True: " + str(true))
    print("False: " + str(false))
    F_score = get_F_score(confusion_matrix)
    print("F score: " + str(F_score))
    return F_score


def vectorize_dataset(dataset):
    for i in range(len(dataset)):
        dataset[i][len(dataset[0]) - 1] = mapping[dataset[i][len(dataset[0]) - 1]]


def one_hot_transformation(dataset):
    new_dataset = dataset.copy()
    for i in range(len(new_dataset)):
        new_dataset[i][len(new_dataset[i]) - 1] = number_to_vector[new_dataset[i][len(new_dataset[i]) - 1]]
    return new_dataset


def find_best_combination(dataset):
    distance_functions = [manhattan, euclidean, chebyshev]
    kernel_functions = [uniform, triangular, epanechnikov, quartic, triweight,
                        tricube, gaussian, cosine, logistic, sigmoid]
    max_F_score = 0
    best_combination = []
    for distance_function in distance_functions:
        for kernel_function in kernel_functions:
            for is_variable in [True, False]:
                if is_variable:
                    windows = [i for i in range(1, 30)]
                else:
                    windows = [0.1 * i for i in range(1, 30)]
                for window in windows:
                    F_score = regression(dataset, distance_function, kernel_function, window, is_variable)
                    if F_score >= max_F_score:
                        max_F_score = F_score
                        best_combination = [distance_function, kernel_function, window]
    print("=====================")
    print("Max F score: " + str(max_F_score))
    print("Best combination: " + str(best_combination))
    return best_combination


def draw_graph(x_values, y_values, x_name, y_name):
    plt.plot(x_values, y_values, 'b')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def build_F_graph(dataset, distance_function, kernel_function, windows, is_variable):
    F_scores = []
    for window in windows:
        F_score = regression(dataset, distance_function, kernel_function, window, is_variable)
        F_scores.append(F_score)
    if is_variable:
        x_name = 'Количество соседей'
    else:
        x_name = 'Ширина окна'
    draw_graph(windows, F_scores, x_name, 'F-мера')


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

number_to_vector = {
    1.0: [1, 0, 0],
    2.0: [0, 1, 0],
    3.0: [0, 0, 1]
}

classes = [1.0, 2.0, 3.0]
filename = '../algorithms/data.csv'
data = pd.read_csv(filename)
dataset = data.values
instances = len(dataset)
vectorize_dataset(dataset)
build_F_graph(dataset, euclidean, tricube, [i for i in range(1, 30)], True)
build_F_graph(dataset, euclidean, tricube, [0.1 * i for i in range(1, 30)], False)
one_hot_dataset = numpy.asarray(one_hot_transformation(dataset))
build_F_graph(one_hot_dataset, manhattan, gaussian, [i for i in range(1, 30)], True)
build_F_graph(one_hot_dataset, manhattan, gaussian, [0.2 * i for i in range(1, 30)], False)

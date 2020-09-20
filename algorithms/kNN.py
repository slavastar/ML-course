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


def predict(dataset, target, distance_function, kernel_function, h):
    sorted_dataset = sorted(dataset, key=lambda row: distance_function(row[0:len(row) - 1], target))
    numerator = 0
    denominator = 0
    for j in range(neighbors):
        kernel_value = kernel_function(j / h)
        numerator += sorted_dataset[j][len(dataset[0]) - 1] * kernel_value
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


def regression(dataset, distance_function, kernel_function, h):
    min_max = minmax(dataset)
    dataset = normalize(dataset, min_max)
    true = false = 0
    confusion_matrix = [[0 for j in range(len(classes))] for i in range(len(classes))]
    for i in range(len(dataset)):
        dataset_train = dataset.copy()
        dataset_test = dataset_train.pop(i)
        prediction = round(predict(dataset_train, dataset_test[0:len(dataset_test) - 1], distance_function, kernel_function, h))
        real = dataset_test[len(dataset_test) - 1]
        confusion_matrix[number_to_index[prediction]][number_to_index[real]] += 1
        if prediction == real:
            true += 1
        else:
            false += 1
    print("----------")
    print("Distance: " + str(distance_function.__name__))
    print("Kernel: " + str(kernel_function.__name__))
    print("h: " + str(h))
    print("True: " + str(true))
    print("False: " + str(false))
    F_score = get_F_score(confusion_matrix)
    print("F score: " + str(F_score))
    return F_score


def vectorize_dataset(dataset):
    for i in range(len(dataset)):
        dataset[i][len(dataset[0]) - 1] = mapping[dataset[i][len(dataset[0]) - 1]]


def one_hot_transformation(dataset):
    new_dataset = []
    for row in dataset:
        new_dataset.append(list(row))
        for i in range(3):
            if i == number_to_index[row[len(dataset[0]) - 1]]:
                value = 1.0
            else:
                value = 0.0
            new_dataset[len(new_dataset) - 1].insert(len(dataset[0]) - 1 + i, value)
    return list(new_dataset)


# best combination: euclidean, triweight, h = 25
def find_best_combination(dataset):
    distance_functions = [manhattan, euclidean]
    kernel_functions = [uniform, triangular, epanechnikov, quartic, triweight,
                        tricube, gaussian, cosine, logistic, sigmoid]
    max_F_score = 0
    best_combination = []
    for distance_function in distance_functions:
        for kernel_function in kernel_functions:
            for h_value in [neighbors]:
                F_score = regression(dataset, distance_function, kernel_function, h_value)
                if F_score >= max_F_score:
                    max_F_score = F_score
                    best_combination = [distance_function, kernel_function, h_value]
    print("=============")
    print("Max F score: " + str(max_F_score))
    print("Best combination: " + str(best_combination))
    return best_combination


def draw_graph(x_values, y_values):
    plt.plot(x_values, y_values, 'ro')
    plt.xlabel("Ширина окна")
    plt.ylabel("F-мера")
    plt.show()


def build_graph(dataset, distance_function, kernel_function):
    h_values = [(neighbors + i) for i in range(0, 20)]
    F_scores = []
    for h_value in h_values:
        F_score = regression(dataset, distance_function, kernel_function, h_value)
        F_scores.append(F_score)
    draw_graph(h_values, F_scores)


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

number_to_list = {
    1.0: [1, 0, 0],
    2.0: [0, 1, 0],
    3.0: [0, 0, 1]
}

neighbors = 25
classes = [1.0, 2.0, 3.0]
filename = 'data.csv'
data = pd.read_csv(filename)
dataset = data.values
instances = len(dataset)
vectorize_dataset(dataset)
build_graph(dataset, euclidean, gaussian)
# new_dataset = numpy.asarray(one_hot_transformation(dataset))
# find_best_combination(new_dataset)

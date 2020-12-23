import csv
from copy import copy

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def read_dataset(filename):
    train = pd.read_csv(filename)
    X = train.iloc[:, :-1].to_numpy()
    Y = train.iloc[:, -1].to_numpy()
    return X, Y


def vectorize(X):
    pass


def vectorize_dataset(X):
    objects = len(X)
    features = len(X[0])
    for i in range(features):
        current_features = 0
        features_mapping = {}
        for j in range(objects):
            if X[i][j]:
                pass


def reorder(in_file, out_file):
    with open(in_file, 'r', newline='') as in_file_handle:
        reader = csv.reader(in_file_handle)
        content = []
        for row in reader:
            content.append(row[1:]+[row[0]])
        with open(out_file, 'w', newline='') as out_file_handle:
            writer = csv.writer(out_file_handle)
            writer.writerows(content)


def normalize(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler.transform(X)


def minkowski_distance(a, b, p):
    result = 0
    for i in range(len(a)):
        result += abs(a[i] - b[i]) ** p
    return result ** (1 / p)


def euclidean_distance(a, b):
    return minkowski_distance(a, b, 2)


def manhattan_distance(a, b):
    return minkowski_distance(a, b, 1)


def find_neighboors(X, j, distance_function, eps):
    N = []
    p = list(X[j])
    for i in range(len(X)):
        x = list(X[i])
        if distance_function(x, p) <= eps:
            N.append(i)
    return N


def DBSCAN(X, distance_function, eps, min_points):
    NOISE = -1
    UNDEFINED = 0
    objects = len(X)
    labels = [UNDEFINED] * objects
    cluster_level = 0
    for i in range(objects):
        if labels[i] != UNDEFINED:
            continue
        N = find_neighboors(X, i, distance_function, eps)
        if len(N) < min_points:
            labels[i] = NOISE
            continue
        cluster_level += 1
        labels[i] = cluster_level
        S = copy(N)
        S.remove(i)
        for k in S:
            if labels[k] == NOISE:
                labels[k] = cluster_level
            if labels[k] != UNDEFINED:
                continue
            labels[k] = cluster_level
            L = find_neighboors(X, k, distance_function, eps)
            if len(L) >= min_points:
                S.extend(L)
    return labels


def rand(Y_real, Y_predict):
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(Y_real)):
        for j in range(len(Y_predict)):
            res1 = Y_real[i] == Y_real[j]
            res2 = Y_predict[i] == Y_predict[j]
            if res1 and res2:
                TP += 1
            elif res1 and not res2:
                FP += 1
            elif not res1 and res2:
                TN += 1
            elif not res1 and not res2:
                FN += 1
    return (TP + FN) / (TP + TN + FP + FN)


colors = {
    -1: "black",
    1: "gold",
    2: "orangered",
    3: "dodgerblue",
    4: "orange",
    5: "red",
    6: "yellow",
    7: "brown",
    8: "magenta",
    9: "deepskyblue",
    10:  "greenyellow"
}


def print_2D(X, Y, title):
    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=12)
    for i in range(len(X)):
        color = colors[Y[i]]
        plt.scatter(X[i][0], X[i][1], marker='o', color=color)
    plt.show()


def print_3D(X, Y, title):
    ax = plt.axes(projection='3d')
    for i in range(len(X)):
        point = X[i]
        color = colors[Y[i]]
        ax.scatter(point[0], point[1], point[2], color=color)
    ax.set_title(title)
    plt.show()


def labels_to_sets(Y):
    sets = []
    for i in range(max(Y) + 1):
        sets.append(set())
    for i in range(len(Y)):
        if Y[i] == -1:
            sets[0].add(i)
        else:
            sets[Y[i]].add(i)
    return sets


def min_separation(X, clusters, distance_function):
    min_distance = 2 ** 31
    for i in range(1, len(clusters)):
        for j in range(1, len(clusters)):
            if i != j:
                for first in clusters[i]:
                    for second in clusters[j]:
                        if distance_function(list(X[first]), list(X[second])) < min_distance:
                            min_distance = distance_function(list(X[first]), list(X[second]))
    return min_distance


def max_diameter(X, clusters, distance_function):
    max_distance = 0
    for i in range(1, len(clusters)):
        for first in clusters[i]:
            for second in clusters[i]:
                if first != second and distance_function(list(X[first]), list(X[second])) > max_distance:
                    max_distance = distance_function(list(X[first]), list(X[second]))
    return max_distance


def dunn_index(X, Y, distance_function):
    clusters = labels_to_sets(Y)
    min_sep = min_separation(X, clusters, distance_function)
    max_diam = max_diameter(X, clusters, distance_function)
    return min_sep / max_diam


def plot(X, Y, title):
    for n in [2, 3]:
        pca = PCA(n_components=n)
        pca.fit(X)
        X_new = pca.transform(X)
        if n == 2:
            print_2D(X_new, Y, title)
        elif n == 3:
            print_3D(X_new, Y, title)


distance_mapping = {
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance
}


filename = "wine.csv"
X, Y_real = read_dataset(filename)
X = normalize(X)
eps = 0.405
min_points = 4
distance = "euclidean"
distance_function = distance_mapping[distance]
Y_predict = DBSCAN(X, distance_function, eps, min_points)
rand_value = rand(Y_real, Y_predict)
dunn_value = dunn_index(X, Y_predict, distance_function)
print("Rand Index: {}".format(round(rand_value, 3)))
print("Dunn Index: {}".format(round(dunn_value, 3)))
print("Number of classes: {}".format(max(Y_predict)))
print("Real:   ", list(Y_real))
print("Predict:", Y_predict)
# plot(X, Y_real, "{}: real".format(filename))
# plot(X, Y_predict, "{}: eps = {}, M = {}, {}".format(filename, eps, min_points, distance))

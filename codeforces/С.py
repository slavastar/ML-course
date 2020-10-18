import math


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


def query(distance_function, kernel_function, window_type, x, y, q, k):
    numerator = 0
    denominator = 0
    for i in range(n):
        if window_type == 'fixed':
            if k == 0:
                kernel_value = 0
            else:
                kernel_value = kernel(kernel_function, distance(distance_function, x[i], q) / k)
        elif window_type == 'variable':
            kernel_value = kernel(kernel_function, distance(distance_function, x[i], q) / distance(distance_function, x[k], q))
        numerator += y[i] * kernel_value
        denominator += kernel_value
    if denominator == 0:
        return 0
    return numerator / denominator


n, m = map(int, input().split())
D = []
x = []
y = []
for i in range(n):
    D.append(list(map(int, input().split())))
q = list(map(int, input().split()))
distance_function = input()
kernel_function = input()
window_type = input()
h = int(input())
D = sorted(D, key=lambda row: distance(distance_function, row[0:m], q))
for i in range(n):
    x.append(D[i][0:m])
    y.append(D[i][m])
result = query(distance_function, kernel_function, window_type, x, y, q, h)
print(result)

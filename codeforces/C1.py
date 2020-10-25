import math

N, M = map(int, input().split())
dataset = list(list())
summ = 0
for x in range(N):
    buff = list(map(int, input().strip().split()))
    dataset.append(buff)
    summ += buff[-1]
common = summ / N
target = list(map(int, input().split()))


def euclidean(x, y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))


def manhattan(x, y):
    return sum([abs(a - b) for a, b in zip(x, y)])


def chebyshev(x, y):
    return max([abs(b - a) for a, b in zip(x, y)])


distances = [euclidean, manhattan, chebyshev]

distance_name = input()
dist = list(filter(lambda x: x.__name__ == distance_name, distances))[0]


def epanechnikov(u):
    return 3 * (1 - pow(u, 2)) / 4 if u < 1 else 0


def uniform(u):
    return 0.5 if u < 1 else 0


def triangular(u):
    return 1 - abs(u) if u < 1 else 0


def quartic(u):
    return 15 * pow(1 - pow(u, 2), 2) / 16 if u < 1 else 0


def triweight(u):
    return 35 * pow(1 - pow(u, 2), 3) / 32 if u < 1 else 0


def tricube(u):
    return 70 * pow(1 - pow(abs(u), 3), 3) / 81 if u < 1 else 0


def gaussian(u):
    return math.exp(-pow(u, 2) / 2) / math.sqrt(2 * math.pi)


def cosine(u):
    return (math.pi * math.cos((math.pi * u) / 2)) / 4 if u < 1 else 0


def logistic(u):
    return 1 / (math.exp(u) + 2 + math.exp(-u))


def sigmoid(u):
    return 2 / (math.pi * (math.exp(u) + math.exp(-u)))


kernels = [epanechnikov, uniform, triangular, quartic, triweight, tricube, gaussian, cosine, logistic, sigmoid]

kernel_name = input()
kernel = list(filter(lambda x: x.__name__ == kernel_name, kernels))[0]

variable = input() == 'variable'

h = int(input())


def kNN(dataset, distf, target, k):
    distances = list()
    for row in dataset:
        dist = distf(target, row[:-1])
        distances.append((row, dist))
    distances.sort(key=lambda tup: tup[1])

    return distances[k][1]


def NWreg(dataset, x, dist, kernel, H, variable):
    up = 0
    gr = 0
    if variable:
        H = kNN(dataset, dist, x, H)
    for row in dataset:
        d = dist(row[:-1], x)
        if d != 0:
            weight = kernel(d / H) if H != 0 else 0
        else:
            weight = kernel(H)
        up += row[-1] * weight
        gr += weight
    return up / gr if gr != 0 else common


print(NWreg(dataset, target, dist, kernel, h, variable))
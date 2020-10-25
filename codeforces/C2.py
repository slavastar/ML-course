
from cmath import sqrt

import math

# distances

def euclideanDistance(first, second):
    distance = 0.0
    for i in range(len(first) - 1):
        distance += (second[i] - first[i]) ** 2
    return sqrt(distance).real


def manhattanDistance(first, second):
    distance = 0.0
    for i in range(len(first) - 1):
        distance += abs(first[i] - second[i])
    return distance


def chebyshevDistance(first, second):
    components = []
    for i in range(len(first) - 1):
        components.append(abs(first[i] - second[i]))
    return max(components)


# Kernels

def uniformKernel(u):
    if abs(u) >= 1:
        return 0
    return 0.5


def triangularKernel(u):
    if abs(u) >= 1:
        return 0
    return 1 - abs(u)


def epanechnikovKernel(u):
    if abs(u) >= 1:
        return 0
    return (3 / 4) * (1 - pow(u, 2))


def quarticKernel(u):
    if abs(u) >= 1:
        return 0
    return (15 / 16) * pow((1 - pow(u, 2)), 2)


def triweightKernel(u):
    if abs(u) >= 1:
        return 0
    return (35 / 32) * pow((1 - pow(u, 2)), 3)


def tricubeKernel(u):
    if abs(u) >= 1:
        return 0
    return (70 / 81) * pow((1 - pow(abs(u), 3)), 3)


def gaussianKernel(u):
    return (1 / sqrt(2 * math.pi)).real * math.exp(-0.5 * pow(u, 2)).real


def cosineKernel(u):
    if abs(u) >= 1:
        return 0
    return (math.pi / 4) * math.cos((math.pi / 2) * u)


def logisticKernel(u):
    return 1 / (math.exp(u) + 2 + math.exp(-u))


def sigmoidKernel(u):
    return (2 / math.pi) * (1 / (math.exp(u) + math.exp(-u)))


# prediction

def getNeighbors(data, target, countOfNeighbors, distanceFunc):
    distances = []
    for row in data:
        dist = distanceFunc(row, target)
        distances.append(dist)
    distances.sort()

    neighbors = []
    for i in range(countOfNeighbors):
        neighbors.append(distances[i])

    return neighbors


def predict(data, target, distanceFunc, kernelFunc, windowDesc, isWindowFixed):

    numerator = 0
    denominator = 0

    currWindow = windowDesc

    if not isWindowFixed:
        neighbors = getNeighbors(data, target, windowDesc + 1, distanceFunc)
        currWindow = neighbors[windowDesc]

    sum = 0
    for neighbor in data:
        distance = distanceFunc(neighbor, target)
        
        sum += neighbor[len(neighbor) - 1]

        kernel = 0
        
        if distance == 0:
            kernel = kernelFunc(currWindow)
        
        if currWindow != 0:
            u = distance / currWindow
            kernel = kernelFunc(u)
        

        numerator += neighbor[len(neighbor) - 1] * kernel
        denominator += kernel
    
    if denominator != 0:
        prediction = numerator / denominator
    else:
        prediction = sum / len(data)

    return prediction


firstRow = input().split()

countOfObjects = int(firstRow[0])
countOfSigns = int(firstRow[1])
data = []

for i in range(countOfObjects):
    row = input().split()
    obj = []
    for value in row:
        obj.append(int(value))
    data.append(obj)

targetRow = input().split()

target = []

for value in targetRow:
    target.append(int(value))

distance = input()
kernel = input()
window = input()
windowDesc = int(input())

currentDistance = euclideanDistance

if distance == "manhattan":
    currentDistance = manhattanDistance
elif distance == "chebyshev":
    currentDistance = chebyshevDistance

currentKernel = uniformKernel

if kernel == "triangular":
    currentKernel = triangularKernel
elif kernel == "epanechnikov":
    currentKernel = epanechnikovKernel
elif kernel == "quartic":
    currentKernel = quarticKernel
elif kernel == "triweight":
    currentKernel = triweightKernel
elif kernel == "tricube":
    currentKernel = tricubeKernel
elif kernel == "gaussian":
    currentKernel = gaussianKernel
elif kernel == "cosine":
    currentKernel = cosineKernel
elif kernel == "logistic":
    currentKernel = logisticKernel
elif kernel == "sigmoid":
    currentKernel = sigmoidKernel

if window == "fixed":
    print(predict(data, target, currentDistance, currentKernel, windowDesc, True))
else:
    print(predict(data, target, currentDistance, currentKernel, windowDesc, False))


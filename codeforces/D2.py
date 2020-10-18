import numpy as np

n, m = map(int, input().split())
X = np.zeros(shape=(n, m))
Y = np.zeros(shape=n)
for i in range(n):
    line = list(map(int, input().split()))
    Y[i] = line.pop(len(line) - 1)
    for j in range(len(line)):
        X[i][j] = line[j]
print(X)
print(Y)

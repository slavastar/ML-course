from collections import defaultdict


x1, x2 = map(int, input().split())
n = int(input())
F, G, V, X, Y = defaultdict(lambda: 0), defaultdict(lambda: 0), defaultdict(lambda: 0), [], []
for i in range(n):
    x, y = map(int, input().split())
    X.append(x)
    Y.append(y)
    F[x] += 1 / n
    G[y] += 1 / n
    V[x, y] += 1
xi = n
for [key, value] in V.items():
    x, y = key[0], key[1]
    result = F[x] * G[y] * n
    xi += (value - result) ** 2 / result - result
print(xi)

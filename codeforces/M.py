from collections import defaultdict
from math import log


kx, ky = map(int, input().split())
n = int(input())
x_prob, y_prob = defaultdict(lambda: 0), defaultdict(lambda: 0)
for i in range(n):
    x, y = map(int, input().split())
    x_prob[x] += 1
    y_prob[x, y] += 1
for key in x_prob:
    x_prob[key] /= n
entropy = 0
for [key, value] in y_prob.items():
    x, y = key[0], key[1]
    value /= n
    entropy -= value * (log(value / x_prob[x]))
print(entropy)

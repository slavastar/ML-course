import math


def mean(lst):
    return sum(lst) / len(lst)


def dispersion(lst):
    mean_lst = mean(lst)
    result = 0
    for element in lst:
        result += (element - mean_lst) ** 2
    return result


def covariation(x, y):
    result = 0
    mean_x = mean(x)
    mean_y = mean(y)
    for i in range(len(x)):
        result += (x[i] - mean_x) * (y[i] - mean_y)
    return result


def pirson_correlation_coefficient(x, y):
    dispersion_x = dispersion(x)
    dispersion_y = dispersion(y)
    return 0 if (dispersion_x == 0 or dispersion_y == 0) else covariation(x, y) / math.sqrt(dispersion(x) * dispersion(y))


n = int(input())
x, y = [], []
for i in range(n):
    cur_x, cur_y = map(int, input().split())
    x.append(cur_x)
    y.append(cur_y)
answer = pirson_correlation_coefficient(x, y)
print(answer)

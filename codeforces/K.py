k, n = int(input()), int(input())
elements = [[] for i in range(k)]
for i in range(n):
    x, y = map(int, input().split())
    elements[x - 1].append(y)
result = 0
for lst in elements:
    if len(lst) != 0:
        mean = sum(lst) / len(lst)
        for element in lst:
            result += (mean - element) ** 2
print(result / n)

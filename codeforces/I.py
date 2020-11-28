def binary_search(x, element, start, end):
    if start > end:
        return -1
    mid = (start + end) // 2
    if element == x[mid]:
        return mid
    if element < x[mid]:
        return binary_search(x, element, start, mid - 1)
    else:
        return binary_search(x, element, mid + 1, end)


def get_ranks(x):
    ranks = []
    sorted_x = sorted(x)
    for element in x:
        ranks.append(binary_search(sorted_x, element, 0, len(x) - 1) + 1)
    return ranks


def spirman_correlation_coefficient(x, y):
    x_ranks, y_ranks = get_ranks(x), get_ranks(y)
    n = len(x)
    result = 0
    for i in range(n):
        result += (x_ranks[i] - y_ranks[i]) ** 2
    result *= - 6 / (n * (n - 1) * (n + 1))
    return result + 1


n = int(input())
x, y = [], []
for i in range(n):
    cur_x, cur_y = map(int, input().split())
    x.append(cur_x)
    y.append(cur_y)
answer = spirman_correlation_coefficient(x, y)
print(answer)

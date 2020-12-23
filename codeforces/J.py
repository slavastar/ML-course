def sum_pairs(elements):
    sum = 0
    k = len(elements)
    for i in range(k - 1, -1, -1):
        sum += i * elements[i] - (k - 1 - i) * elements[i]
    return sum


k, n = int(input()), int(input())
elements = [[] for i in range(k)]
for i in range(n):
    a, b = map(int, input().split())
    elements[b - 1].append(a)
for i in range(len(elements)):
    elements[i] = sorted(elements[i])
internal_sum = 2 * sum([sum_pairs(elements_list) for elements_list in elements])
external_sum = 2 * sum_pairs(sorted([value for sublist in elements for value in sublist])) - internal_sum
print(internal_sum, "\n", external_sum, sep='')

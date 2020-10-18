def transpone(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def harmonic_mean(a, b):
    return divide(2 * a * b, (a + b))

def divide(a, b):
    if b == 0:
        return 0
    return a / b

k = int(input())
matrix = []
for i in range(k):
    matrix.append(list(map(int, input().split())))
matrix = transpone(matrix)

C = []
P = []
TP = []
FP = []
FN = []
precision = []
recall = []
F = []
all = 0
micro_F = 0
precision_w = 0
recall_w = 0
for i in range(k):
    TP.append(matrix[i][i])
    row = matrix[i]
    column = []
    for j in range(k):
        column.append(matrix[j][i])
    sum_row = sum(row)
    sum_column = sum(column)
    P.append(sum_row)
    C.append(sum_column)
    FP.append(sum_row - matrix[i][i])
    FN.append(sum_column - matrix[i][i])
    precision.append(divide(TP[i], (TP[i] + FP[i])))
    recall.append(divide(TP[i], (TP[i] + FN[i])))
    F.append(harmonic_mean(precision[i], recall[i]))
    micro_F += C[i] * F[i]
    precision_w += divide(TP[i] * C[i], P[i])
    recall_w += TP[i]
all = sum(C)
micro_F /= all
precision_w /= all
recall_w /= all
macro_F = harmonic_mean(precision_w, recall_w)
print('P: ' + str(P))
print('C: ' + str(C))
print('TP: ' + str(TP))
print('FP: ' + str(FP))
print('FN: ' + str(FN))
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
print('F: ' + str(F))
print('All: ' + str(all))
print('Precision_w: ' + str(precision_w))
print('Recall_w: ' + str(recall_w))
print('Micro_F: ' + str(micro_F))
print('Macro_F: ' + str(macro_F))
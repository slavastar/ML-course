from math import e, log2, log
import random


def H(M):
    return max(-M, 0)


def H_derivative(M):
    if M < 0:
        return -1
    return 0


def V(M):
    return max(1 - M, 0)


def V_derivative(M):
    if M > 1:
        return -1
    return 0


def L(M):
    return log2(1 + e ** (-M))


def L_derivative(M):
    return -(e ** (-M)) / ((1 + e ** (-M)) * log(2))


def Q(M):
    return (1 - M) ** 2


def Q_derivative(M):
    return 2 * (M - 1)


def S(M):
    return 2 / (1 + e ** M)


def S_derivative(M):
    return -2 * (1 + e ** M) ** -2 * e ** M


def E(M):
    return e ** (-M)


def E_derivative(M):
    return - (e ** (-M))


def scalar_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def subtract(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i] - b[i])
    return c


def multiply(const, a):
    return [const * component for component in a]


def margin(w, x, y):
    return scalar_product(w, x) * y


def initialize_weights(option):
    if option == 1:
        return [random.uniform(-0.5 / m, 0.5 / m) for i in range(m + 1)]
    elif option == 2:
        return [0 for i in range(m + 1)]


def initialize_loss_value(w, X, Y, loss_function):
    result = 0
    for i in range(n):
        result += loss_function(margin(w, X[i], Y[i]))
    return result / n


# w, x - vector
# y, h - number
def gradient_step(w, x, y, loss_function, h):
    return subtract(w, multiply(h * derivative[loss_function](margin(w, x, y)) * y, x))


def stochastic_gradient(X, Y, loss_function):
    w = initialize_weights(1)
    q = initialize_loss_value(w, X, Y, loss_function)
    iterations = 0
    while iterations < 2000:
        c = 3
        print("Iteration =", iterations, "\tq =", round(q, c), "\t w =", [round(element, c) for element in w])
        iterations += 1
        h = 1 / iterations
        i = random.randint(0, len(X) - 1)
        x, y = X[i], Y[i]
        eps = loss_function(margin(w, x, y))
        w = gradient_step(w, x, y, loss_function, h)
        alpha = 1 / 2
        q_next = (1 - alpha) * q + alpha * eps
        delta_q = abs(q - q_next)
        if delta_q < 0.03:
            break
        q = q_next
    return w


derivative = {
    H: H_derivative,
    V: V_derivative,
    L: L_derivative,
    Q: Q_derivative,
    S: S_derivative,
    E: E_derivative
}


n, m = map(int, input().split())
X = []
Y = []
for i in range(n):
    line = [1]
    line.extend(list(map(int, input().split())))
    Y.append(line.pop(len(line) - 1))
    X.append(line)
w = stochastic_gradient(X, Y, S)
print(w)
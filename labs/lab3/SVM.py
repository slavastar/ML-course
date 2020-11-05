import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score


def print_points(X, Y):
    for i in range(len(X)):
        color = 'blue' if Y[i] == 1 else 'green'
        plt.scatter(X[i][0], X[i][1], marker='o', color=color)


def kernel_linear(x, y):
    return np.dot(x, y)


def kernel_polynomial(x, y, degree):
    return np.power(1 + np.dot(x, y), degree)


def kernel_gaussian(x, y, betta):
    return np.exp(-betta * np.power(np.linalg.norm(x - y), 2))


def get_kernel_function(name, degree=2, betta=1):
    if name == "linear":
        return lambda x, y: kernel_linear(x, y)
    elif name == "polynomial":
        return lambda x, y: kernel_polynomial(x, y, degree)
    else:
        return lambda x, y: kernel_gaussian(x, y, betta)


def find_score(X, Y, kernel_function, C):
    splits = 4
    accuracy = 0
    kf = KFold(n_splits=splits)
    for train_indices, test_indices in kf.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        svm = SVM(X_train, Y_train, C, kernel_function)
        classifier = svm.classifier
        predictions = np.zeros(len(X_test))
        for i in range(len(X_test)):
            predictions[i] = classifier(X_test[i])
        accuracy += accuracy_score(predictions, Y_test)
    accuracy /= splits
    return accuracy


def read_dataset(filename):
    data = pd.read_csv(filename)
    X = data.values[:, :-1]
    Y = np.vectorize(lambda y: -1 if y == 'N' else 1)(data.values[:, -1])
    return X, Y


def calculate_kernel_matrix(kernel_function, X, Y):
    return np.fromfunction(np.vectorize(lambda i, j: kernel_function(X[i], Y[j])), (X.shape[0], Y.shape[0]), dtype=int)


class SVM:

    def __init__(self, X, Y, C, kernel_function):
        self.X = X
        self.Y = Y
        self.C = C
        self.n = len(X)
        self.kernel_function = kernel_function
        self.K = self.calculate_kernel_matrix()
        self.alpha = self.find_alpha()
        self.w = self.find_w()
        self.b = self.find_b()
        self.classifier = self.find_classifier()

    def calculate_kernel_matrix(self):
        n, *_ = X.shape
        m, *_ = Y.shape
        f = lambda i, j: self.kernel_function(X[i], X[j])
        return np.fromfunction(np.vectorize(f), (n, m), dtype=int)

    def find_alpha(self):

        def objective(alpha):
            result = 0
            for i in range(self.n):
                for j in range(self.n):
                    result += alpha[i] * alpha[j] * self.Y[i] * self.Y[j] * self.K[i][j]
            result *= 0.5
            result -= np.sum(alpha)
            return result

        def constraint(alpha):
            return np.sum(alpha * self.Y)

        b = (0, self.C)
        bounds = tuple([b] * self.n)
        alpha = np.full(self.n, self.C / 2)
        constraints = ([{'type': 'eq', 'fun': constraint}])

        solution = minimize(objective, alpha, method='SLSQP', \
                            bounds=bounds, constraints=constraints, tol=1)

        alpha = solution.x
        return alpha

    def find_w(self):
        w = np.zeros(len(self.X[0]))
        for i in range(self.n):
            w = np.add(w, np.dot(self.alpha[i] * self.Y[i], self.X[i]))
        return w

    def find_b(self):
        b = 0
        for i in range(self.n):
            b += self.kernel_function(self.w, X[i]) - Y[i]
        return b / self.n

    def find_classifier(self):
        def classifier(q):
            kernel = calculate_kernel_matrix(self.kernel_function, np.array([q]), X)
            result = -self.b
            for i in range(self.n):
                result += self.alpha[i] * self.Y[i] * kernel[0][i]
            return int(np.sign(result))

        return classifier


def SGD(X, Y, kernel_function, C):
    n = len(X)
    alpha = np.full(n, C / 1000)
    iteration_limit = 500
    K = calculate_kernel_matrix(kernel_function, X, X)
    iterations = 0
    real_iterations = 0
    while iterations < iteration_limit:
        k = iterations % n
        iterations += 1
        learning_rate = 10 / iterations
        alpha_prev = alpha.copy()
        l = np.ones(n)
        for i in range(n - 1):
            for j in range(n):
                l[i] -= alpha[j] * Y[k] * Y[j] * K[k][i]
            alpha[i] += learning_rate * l[i]
            if alpha[i] < 0 or alpha[i] > C:
                alpha = alpha_prev.copy()
                continue
        value = 0
        for i in range(n - 1):
            value += alpha[i] * Y[i]
        alpha[n - 1] = - value / Y[n - 1]
        if alpha[n - 1] < 0 or alpha[n - 1] > C:
            alpha = alpha_prev.copy()
            continue
        real_iterations += 1

    print("real iterations:", real_iterations)
    result = 0
    for i in range(n):
        for j in range(n):
            result -= alpha[i] * alpha[j] * Y[i] * Y[j] * K[i][j]
    result *= -0.5
    result += np.sum(alpha)
    print(result)

    # find w
    w = np.zeros(len(X[0]))
    for i in range(n):
        w = np.add(w, np.dot(alpha[i] * Y[i], X[i]))

    # find b
    b = 0
    for i in range(n):
        b += kernel_function(w, X[i]) - Y[i]
    b /= n

    # find classifier
    def classifier(q):
        kernel = calculate_kernel_matrix(kernel_function, np.array([q]), X)
        result = -b
        for i in range(n):
            result += alpha[i] * Y[i] * kernel[0][i]
        return int(np.sign(result))

    print(alpha)

    return classifier


def draw(classifier, X, Y):
    d = 50
    x_min, y_min = np.amin(X, 0)
    x_max, y_max = np.amax(X, 0)
    x_delta, y_delta = (x_max - x_min) / d, (y_max - y_min) / d
    x_min -= x_delta
    x_max += x_delta
    y_min -= y_delta
    y_max += y_delta
    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_delta), np.arange(y_min, y_max, y_delta))
    points = np.c_[xx.ravel(), yy.ravel()]
    for point in points:
        color = "#e71e52" if classifier(point) == 1 else "orange"
        plt.scatter(point[0], point[1], s=120, color=color)
    print_points(X, Y)
    plt.show()


def find_best_hyperparameters(X, Y):
    C_values = [0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    round_digits = 3

    # linear
    print("Linear")
    best_linear_score = 0
    best_linear_hyperparameters = [0]
    kernel_function = get_kernel_function("linear")
    for C in C_values:
        score = find_score(X, Y, kernel_function, C)
        print("Score =", round(score, round_digits), "\tC =", C)
        if score >= best_linear_score:
            best_linear_score = score
            best_linear_hyperparameters = [C]

    print("Best linear score:", round(best_linear_score, round_digits))
    print("Best linear hyperparameters: C =", best_linear_hyperparameters[0])

    # polynomial
    print("Polynomial")
    best_polynomial_score = 0
    best_polynomial_hyperparameters = [0, 0]
    degree_values = [2, 3, 4, 5]
    for degree in degree_values:
        kernel_function = get_kernel_function("polynomial", degree=degree)
        for C in C_values:
            score = find_score(X, Y, kernel_function, C)
            print("Score =", round(score, round_digits), "\tC =", C, "\tdegree =", degree)
            if score >= best_polynomial_score:
                best_polynomial_score = score
                best_polynomial_hyperparameters = [C, degree]

    print("Best polynomial score:", round(best_polynomial_score, round_digits))
    print("Best polynomial hyperparameters: C =",
          best_polynomial_hyperparameters[0], "\tdegree =", best_polynomial_hyperparameters[1])

    # gaussian
    print("Gaussian")
    best_gaussian_score = 0
    best_gaussian_hyperparameters = [0, 0]
    betta_values = [1, 2, 3, 4, 5]
    for betta in betta_values:
        kernel_function = get_kernel_function("gaussian", betta=betta)
        for C in C_values:
            score = find_score(X, Y, kernel_function, C)
            print("Score =", round(score, round_digits), "\tC =", C, "\tbetta =", betta)
            if score >= best_gaussian_score:
                best_gaussian_score = score
                best_gaussian_hyperparameters = [C, betta]

    print("Best gaussian score:", round(best_gaussian_score, round_digits))
    print("Best gaussian hyperparameters: C =",
          best_gaussian_hyperparameters[0], "\tbetta =", best_gaussian_hyperparameters[1])

    return best_linear_hyperparameters, best_polynomial_hyperparameters, best_gaussian_hyperparameters


X, Y = read_dataset("datasets/chips.csv")

# kernel_function = get_kernel_function("linear")
# clf = SGD(X, Y, kernel_function, 50)
# draw(clf, X, Y)

kernel_function = get_kernel_function("linear")
svm = SVM(X, Y, 50, kernel_function)
draw(svm.classifier, X, Y)

kernel_function = get_kernel_function("polynomial", degree=2)
svm = SVM(X, Y, 10, kernel_function)
draw(svm.classifier, X, Y)

kernel_function = get_kernel_function("gaussian", betta=5)
svm = SVM(X, Y, 0.5, kernel_function)
draw(svm.classifier, X, Y)


"""
chips.csv
linear: score = 0.667 (C = 5-100)
polynomial: score = 0.725 (C = 5-100, degree = 2), (0.725, 1, 4)
gaussian: score = ? (C = 0.5, betta = 5)

geyser.csv
linear: score = (C = )
polynomial: score = (C = , degree = )
gaussian: score = (C = , betta = )
"""

def f(x):
    return 10 * x ** 2;


def f_derivative(x):
    return 20 * x


def gradient_descent(h=0.01, eps=0.001):
    x_prev = 100
    x = 50
    for i in range(1000):
        print("Step: ", i, "\tx = ", round(x, 5), "\tf(x) = ", round(f(x), 5))
        if abs(x - x_prev) < eps:
            break
        x_prev = x
        x = x - h * f_derivative(x_prev)
    return x


gradient_descent()
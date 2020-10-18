import numpy


def f(x):
    return 10 * x[0] ** 2 + x[1] ** 2


def gradient_descent(h=0.01, eps=0.001):
    x_prev = numpy.array([100, 100])
    x = numpy.array([50, 50])
    for i in range(1000):
        print("Step: ", i, "\tx = ", numpy.round(x, 5), "\tf(x) = ", numpy.round(f(x), 5))
        if numpy.sum((x - x_prev) ** 2) < eps ** 2:
            break
        x_prev = x
        x = x_prev - h * numpy.array(20 * x_prev[0], 2 * x_prev[1])
    return x


gradient_descent()

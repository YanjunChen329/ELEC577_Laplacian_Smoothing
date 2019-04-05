import math
import numpy as np
import matplotlib.pyplot as plt
import copy

R = 1
BETA = math.sqrt(500)


def old_DHF(x, y, z=2.34, r=1, beta=1./math.sqrt(500)):
    exp_one = math.exp(-(x - math.pi)**2 - (y - math.pi)**2 - (z - math.pi)**2)
    part_one = -4 * exp_one

    sum_exp = 0
    for i in range(13):
        x_sqr = (x - r * math.sin(i/2.) - math.pi)**2
        y_sqr = (y - r * math.cos(i/2.) - math.pi)**2
        sum_exp += math.exp(-beta * (x_sqr + y_sqr))
    part_two = 4 * math.cos(x) * math.cos(y) * sum_exp

    return part_one - part_two


def drill_hole_function(x, y, z=2.34, r=R, beta=BETA):
    exp_one = math.exp(-(x - math.pi)**2 - (y - math.pi)**2 - (z - math.pi)**2)
    part_one = -4 * exp_one
    # part_one = 0

    sum_exp = 0
    for i in range(13):
        x_sqr = (x - r * math.sin(i/2.) - math.pi)**2
        y_sqr = (y - r * math.cos(i/2.) - math.pi)**2
        sum_exp += math.exp(-beta * (x_sqr + y_sqr))
    part_two = sum_exp

    return part_one - part_two


def DHF_gradient(x, y, z=2.34, r=R, beta=BETA):
    exp_one = math.exp(-(x - math.pi)**2 - (y - math.pi)**2 - (z - math.pi)**2)
    dx = -4 * exp_one * (-2 * (x - math.pi))
    dy = -4 * exp_one * (-2 * (y - math.pi))

    for i in range(13):
        x_part = (x - r * math.sin(i / 2.) - math.pi)
        y_part = (y - r * math.cos(i / 2.) - math.pi)
        exp_two = 2 * math.exp(-beta * (x_part**2 + y_part**2))
        dx += (beta * 2 * x_part) * exp_two
        dy += (beta * 2 * y_part) * exp_two

    return np.array([dx, dy])


def plot_DHF(dhf_func, load=True):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(1.2, 5, 0.005)
    Y = np.arange(1.2, 5, 0.005)
    X, Y = np.meshgrid(X, Y)

    if not load:
        F = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                F[i, j] = dhf_func(X[i, j], Y[i, j], 2.34)
        np.savetxt("DHF_grid.txt", F)
    else:
        F = np.loadtxt("DHF_grid.txt")

    ax.plot_surface(X, Y, F, cmap='binary')
    x, y = random_start()
    ax.scatter(x, y, dhf_func(x, y, 2.34), c="red", s=100)

    plt.show()


def plot_DHF_contour(start, gd_trace, load=True):
    fig, ax = plt.subplots()
    X = np.arange(1.2, 5, 0.005)
    Y = np.arange(1.2, 5, 0.005)
    X, Y = np.meshgrid(X, Y)

    if not load:
        F = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                F[i, j] = drill_hole_function(X[i, j], Y[i, j], 2.34)
        np.savetxt("DHF_grid.txt", F)
    else:
        F = np.loadtxt("DHF_grid.txt")

    ax.contour(X, Y, F)
    ax.scatter(start[0], start[1], s=100, facecolors='none', edgecolors='r')
    ax.scatter(gd_trace[1:, 0], gd_trace[1:, 1], s=60, facecolors='none', edgecolors='black')
    ax.plot(gd_trace[1:, 0], gd_trace[1:, 1], c='black')
    plt.show()


def random_start(r=1, radius=0.2):
    i = np.random.randint(0, 13)

    x = r * math.sin(i / 2.) + math.pi
    y = r * math.cos(i / 2.) + math.pi

    # r = np.random.uniform(0, radius)
    r = radius
    theta = np.random.uniform(0, 2*math.pi)
    x_offset = r * math.cos(theta)
    y_offset = r * math.sin(theta)

    return np.array([x + x_offset, y + y_offset])


def gradient_descent(start, iteration, stepsize=0.005):
    w = np.array(start)
    old_w = np.zeros(w.shape)
    print(w)
    trace = []

    for k in range(iteration+1):
        w -= stepsize * DHF_gradient(w[0], w[1])
        if k % 2 == 0:
            print(k, w)
            diff = np.linalg.norm(w - old_w)
            if diff < 0.0001:
                break
            old_w = np.array(w)
            trace.append(np.array(w))

    print(trace)
    return np.array(trace)


if __name__ == '__main__':
    # plot_DHF(drill_hole_function)
    start_point = random_start()
    gd_trace = gradient_descent(start_point, 1000)
    print(start_point)
    plot_DHF_contour(start_point, gd_trace)



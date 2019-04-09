import numpy as np
import math
from matplotlib import pyplot as plt
from LaplacianSmoothing import LS_grad_descent

def original_loss(x):
    loss = 0.
    if not x.shape[0] == 100:
        print("X dimension error.")
        return -1
    for i in range(50):
        loss += x[2 * i] ** 2 + x[2 * i + 1] ** 2 / 100.
    return loss


def original_grad(x, noise=0.0):
    grad = np.copy(x)
    for i in range(50):
        grad[2 * i] *= 2.
        grad[2 * i + 1] /= 50.
    grad += np.random.multivariate_normal(np.zeros(100), np.identity(100)) * noise
    return grad


def index_loss(x, idxs):
    loss = 0.
    for i in range(100):
        val = x[i] ** 2
        if i in idxs:
            val /= 100.
        loss += val
    return loss


def index_grad(x, idxs, noise=0.0):
    grad = np.copy(x)
    for i in range(100):
        if i in idxs:
            grad[i] /= 50.
        else:
            grad[i] *=2.
    grad += np.random.multivariate_normal(np.zeros(100), np.identity(100)) * noise
    return grad


def stepsize_decay(init, it):
    decay = math.floor(it / 1000.)
    return init / 10 ** decay

ndim = 100
epsilon = 0.05
rand_idx = np.random.choice(list(range(100)), size=50)
loss = lambda x: index_loss(x, rand_idx)
loss_grad = lambda x: index_grad(x, rand_idx, noise=epsilon)
stepsize1 = lambda i: stepsize_decay(0.1, i)
stepsize2 = lambda i: stepsize_decay(0.1, i)
sigma = 10.
iter_num = 100000
init_x = np.ones(ndim)
loss_gd, dists = LS_grad_descent(init_x, loss, loss_grad, stepsize1, iter_num, sigma, 0, ndim, cal_dist=False)
init_x = np.ones(ndim)
loss_order_1, dists = LS_grad_descent(init_x, loss, loss_grad, stepsize2, iter_num, sigma, 1, ndim, cal_dist=False)
init_x = np.ones(ndim)
loss_order_2, dists = LS_grad_descent(init_x, loss, loss_grad, stepsize2, iter_num, sigma, 2, ndim, cal_dist=False)
iters = list(range(iter_num))
plt.plot(iters, loss_gd, label="GD")
plt.plot(iters, loss_order_1, label="Order1")
plt.plot(iters, loss_order_2, label="Order2")
plt.xscale('log')
plt.yscale('log')
plt.xlim([10, 1e5])
plt.ylim([1e-5, 1e2])
plt.legend()
plt.grid()
plt.xlabel('iterations')
plt.ylabel('|f(x) - f(x*)|')
plt.savefig("RandIdx_LSGD_decay_ep05.png")





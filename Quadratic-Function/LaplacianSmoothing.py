import numpy as np
import matplotlib.pyplot as plt
import argparse
from numpy.fft import fft, ifft

def make_vec(order, ndim):
    if order < 1:
        print("Order not valid.")
        return np.zeros(ndim)
    
    Mat = np.zeros(shape=(order, 2*order+1))
    Mat[0, order-1] = 1.
    Mat[0, order] = -2.
    Mat[0, order+1] = 1.

    for i in range(1, order):
        Mat[i, order-i-1] = 1.
        Mat[i, order+i+1] = 1.
        Mat[i, order] = Mat[i-1, order-1] - 2*Mat[i-1, order] + Mat[i-1, order+1]
        
        Mat[i, order-i] = -2*Mat[i-1, order-i] + Mat[i-1, order-i+1]
        Mat[i, order+i] = Mat[i, order-i]
        
        for j in range(0, i-1):
            Mat[i, order-j-1] = Mat[i-1, order-j-2] - 2*Mat[i-1, order-j-1] + Mat[i-1, order-j]
            Mat[i, order+j+1] = Mat[i, order-j-1]
    
    vec = np.zeros(shape=(1, ndim))
    for i in range(order+1):
        vec[0, i] = Mat[-1, order-i]
    for i in range(order):
        vec[0, -1-i] = Mat[-1, order-i-1]
    return vec


def ls_grad(loss_grad, ndim, order, sigma):
    vec = make_vec(order, ndim)
    coef = (-1) ** order * sigma
    denom = 1 + coef * fft(vec)
    return (lambda x: np.squeeze(np.real(ifft(fft(x) / denom))))


def LS_grad_descent(init_x, loss, loss_grad, stepsize, iter_num, sigma, order, ndim, cal_dist=False, optimal_x=0):
    grad_func = ls_grad(loss_grad, ndim, order, sigma)
    x = init_x
    losses = []
    dists = []
    for i in range(iter_num):
        losses.append(loss(x))
        dists.append(np.linalg.norm(x - optimal_x))
        x -= stepsize(i) * grad_func(x)
    return losses, dists

ndim = 3
a = np.random.rand(ndim)
loss = lambda x: np.linalg.norm(x - a) ** 2
loss_grad = lambda x: x
stepsize = lambda i: 0.5
sigma = 1
order = 2
iter_num = 100
init_x = np.random.rand(ndim)
x_loss, dists = LS_grad_descent(init_x, loss, loss_grad, stepsize, iter_num, sigma, order, ndim, cal_dist=True, optimal_x=a)





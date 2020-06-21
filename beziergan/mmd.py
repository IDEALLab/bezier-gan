"""
Compute the maximum mean discrepancy (MMD) of the generative distribution and the data distribution

Author(s): Wei Chen (wchen459@umd.edu)
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from utils import mean_err


def gaussian_kernel(X, Y, sigma=1.0):
    beta = 1. / (2. * sigma**2)
    dist = pairwise_distances(X, Y)
    s = beta * dist.flatten()
    return np.exp(-s)

def maximum_mean_discrepancy(gen_func, X_test):
    
    X_gen = gen_func(2000)
    X_gen = X_gen.reshape((X_gen.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
      
    mmd = np.mean(gaussian_kernel(X_gen, X_gen)) - \
            2 * np.mean(gaussian_kernel(X_gen, X_test)) + \
            np.mean(gaussian_kernel(X_test, X_test))
    
    return mmd
    
def ci_mmd(n, gen_func, X_test):
    mmds = np.zeros(n)
    for i in range(n):
        mmds[i] = maximum_mean_discrepancy(gen_func, np.squeeze(X_test))
    mean, err = mean_err(mmds)
    return mean, err
"""
Plot generated airfoils and curvatures.

Author(s): Wei Chen (wchen459@umd.edu)
"""

#import numpy as np
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
import tensorflow as tf
from scipy.special import binom

import sys 
sys.path.append('..')
import functions


def plot_airfoil(position, airfoil, P=None, W=None):
    plt.subplot(position)
    plt.plot(airfoil[:,0], airfoil[:,1], 'k-', lw=2, alpha=.5)
    if P is not None and W is not None:
        plt.plot(P[:,0], P[:,1], 'k-', lw=1, alpha=.5)
        plt.scatter(P[:,0], P[:,1], marker='s', c='k', s=10+1000*W, alpha=.5)
    plt.ylim([-.2, .2])
    plt.axis('equal')
    
def plot_curvature(position, tt, curvature):
    plt.subplot(position)
    plt.plot(tt, curvature)
    plt.xlabel(r'$t$')
    plt.ylabel('Curvature')
    # plt.yscale('log')
    
def bernstein(t, i, n):
    b = binom(n, i) * t**i * (1-t)**(n-i)
    return b

def eval_bezier(t, P, W):
    assert P.shape[0] == W.shape[0]
    n = P.shape[0]-1
    numerator = 0
    denominator = 0
    for i in range(n+1):
        numerator += bernstein(t, i, n) * P[i] * W[i]
        denominator += bernstein(t, i, n) * W[i]
    return numerator/denominator

def eval_1st_derivative(t, P, W):
    assert P.shape[0] == W.shape[0]
    fun_x = lambda t: eval_bezier(t, P, W)[0]
    fun_y = lambda t: eval_bezier(t, P, W)[1]
    dx = grad(fun_x)(t)
    dy = grad(fun_y)(t)
    return np.array([dx, dy])

def eval_2nd_derivative(t, P, W):
    assert P.shape[0] == W.shape[0]
    fun_x = lambda t: eval_1st_derivative(t, P, W)[0]
    fun_y = lambda t: eval_1st_derivative(t, P, W)[1]
    ddx = grad(fun_x)(t)
    ddy = grad(fun_y)(t)
    return np.array([ddx, ddy])

def compute_curvature(P, W, resolution=500):
    tt = np.linspace(0., 1., resolution)
    airfoil = np.array([eval_bezier(t, P, W) for t in tt])
    d_airfoil = np.array([eval_1st_derivative(t, P, W) for t in tt])
    dd_airfoil = np.array([eval_2nd_derivative(t, P, W) for t in tt])
    dx = d_airfoil[:,0]
    dy = d_airfoil[:,1]
    ddx = dd_airfoil[:,0]
    ddy = dd_airfoil[:,1]
    cv = np.abs(ddx*dy - dx*ddy)/(dx*dx + dy*dy)**1.5
#    plt.figure()
#    plt.plot(tt, dd_airfoil[:,0], label='ddx')
#    plt.plot(tt, dd_airfoil[:,1], label='ddy')
#    plt.legend(frameon=False)
#    plt.show()
    return tt, cv, airfoil
    

if __name__ == "__main__":
    
    ''' BezierGAN '''
    latent_dim = 4 
    noise_dim = 10
    
    i = 1#np.random.choice(10)
    model_directory = './trained_gan/{}_{}/{}'.format(latent_dim, noise_dim, i)
    tf.keras.backend.clear_session()
    func = functions.AirfoilGAN(latent_dim, noise_dim, model_directory, full=True)
    
    n_airfoils = 3
    alphas = func.sample_design_variables(n_airfoils, 'lhs')
    
    plt.figure(figsize=(n_airfoils*8., 2*4.))
    grid = plt.GridSpec(2, n_airfoils)
    for j in range(n_airfoils):
        alpha = alphas[j]
        airfoil, P, W = func.gan.synthesize(alpha[:latent_dim].reshape(1,-1), alpha[latent_dim:].reshape(1,-1), return_cp=True)
        tt, curvature, airfoil = compute_curvature(P, W)
        plot_airfoil(grid[0,j], airfoil, P, W)
        plot_curvature(grid[1,j], tt, curvature)
    plt.tight_layout()
    plt.savefig('./trained_gan/curvature.svg')
    plt.savefig('./trained_gan/curvature.pdf')
    plt.close()
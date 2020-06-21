"""
Customized Gaussian processes regression that computes delta
References:
-----------
http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier
Author(s): Wei Chen (wchen459@umd.edu)
"""

from __future__ import division
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.gaussian_process.gpr import GaussianProcessRegressor


class GPR(GaussianProcessRegressor):
    
    def _k_nl_tau(self, tau, lambda_val='min'):
        _ = np.zeros((1, self.X_train_.shape[1]))
        if self._K_inv is None:
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            self._K_inv = L_inv.dot(L_inv.T)
        lambdas = np.linalg.eigh(self._K_inv)[0]
        lambda_min = lambdas[0]
        lambda_max = lambdas[-1]
        N = self.X_train_.shape[0]
        if lambda_val == 'min':
            _lambda = lambda_min
        elif lambda_val == 'max':
            _lambda = lambda_max
        else:
            _lambda = .5*(lambda_min+lambda_max)
        k_nl_tau = ((1-tau)*self.kernel_(_, _)/N/_lambda)**0.5
        return k_nl_tau
    
    def _r(self, k_nl_tau):
        if 'RBF' in str(self.kernel_):
            C = self.kernel_.get_params()['k1__constant_value']
            ls = self.kernel_.get_params()['k2__length_scale']
            r = ls * (-2 * np.log(k_nl_tau/C))**0.5
        else:
            pass
        return r.flatten()[0]
    
    def get_r(self, tau, lambda_val='min'):
        k_nl_tau = self._k_nl_tau(tau, lambda_val)
        r = self._r(k_nl_tau)
        return r
    
    def grad_predict(self, x, compute_std=False):
        C = self.kernel_.get_params()['k1__constant_value']
        ls = self.kernel_.get_params()['k2__length_scale']
        if self._K_inv is None:
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            self._K_inv = L_inv.dot(L_inv.T)
        dim = self.X_train_.shape[1]
        x = x.reshape(-1, 1, dim)
        N = self.X_train_.shape[0]
        x = np.tile(x, (1, N, 1)) # M x N x d
        dx = x - self.X_train_ # M x N x d
        kx = C * np.exp(-.5 * np.diagonal(np.matmul(dx, np.transpose(dx,axes=[0,2,1])), axis1=1, axis2=2)/ls**2) # M x N
        kx = np.expand_dims(kx, axis=-1) # M x N x 1
        kx_tile = np.tile(kx, (1, 1, dim)) # M x N x d
        # Gradient of predictive mean
        y_mean_prime = - np.matmul(np.matmul(np.transpose(dx*kx_tile,axes=[0,2,1]), self._K_inv), 
                                   self.y_train_.reshape(-1,1)) / ls**2 # M x d x 1
        y_mean_prime = np.squeeze(y_mean_prime, axis=-1) # M x d
        if compute_std:
            # Gradient of predictive std
            _ = np.zeros((1, dim))
            y_std = (self.kernel_(_, _) - np.matmul(np.matmul(np.transpose(kx,axes=[0,2,1]), self._K_inv), 
                                                    kx))**.5 # M x 1 x 1
            y_std_prime = np.matmul(np.matmul(np.transpose(dx*kx_tile,axes=[0,2,1]), self._K_inv), 
                                    kx) / y_std / ls**2 # M x d x 1
            y_std_prime = np.squeeze(y_std_prime, axis=-1)
            return y_mean_prime, y_std_prime
        else:
            return y_mean_prime

    def k_star(self, tau):
        if self._K_inv is None:
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            self._K_inv = L_inv.dot(L_inv.T)
        w, v = np.linalg.eigh(self._K_inv)
        self.B = np.dot(np.diag(w**.5), v.T)
        _ = np.zeros((1, self.X_train_.shape[1]))
        k_star = ((1-tau)*self.kernel_(_, _))**.5/np.linalg.norm(np.dot(self.B,self.y_train_.reshape(-1,1))) * self.y_train_
        return k_star

    def k_distance(self, x, tau):
        k_star = self.k_star(tau)
        kx = self.kernel_(x, self.X_train_)
        _ = np.zeros((1, self.X_train_.shape[1]))
        return .5*np.linalg.norm(kx-k_star, axis=1)**2 + \
                .5*(np.linalg.norm(np.dot(kx, self.B.T), axis=1)-((1-tau)*self.kernel_(_, _))**.5)**2
        
        
        
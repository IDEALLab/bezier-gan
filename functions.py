#import numpy as np
import autograd.numpy as np
from autograd import grad
import tensorflow as tf
from pyDOE import lhs
from scipy.optimize import minimize

from simulation import evaluate
from beziergan.gan import GAN
from ffd.synthesis import synthesize as synthesize_ffd

from utils import train_test_plit


class Airfoil(object):
    
    def __init__(self):
        self.y = None
        self.bounds = None
        self.dim = None
        self.config_fname = None
            
    def __call__(self, x):
        x = np.array(x, ndmin=2)
        y = np.apply_along_axis(lambda x: evaluate(self.synthesize(x), self.config_fname), 1, x)
        self.y = np.squeeze(y)
        return self.y
    
    def is_feasible(self, x):
        x = np.array(x, ndmin=2)
        if self.y is None:
            self.y = self.__call__(x)
        feasibility = np.logical_not(np.isnan(self.y))
        return feasibility
    
    def synthesize(self, x):
        pass
    
    def sample_design_variables(self, n_sample, method='random'):
        if method == 'lhs':
            x = lhs(self.dim, samples=n_sample, criterion='cm')
            x = x * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
        else:
            x = np.random.uniform(self.bounds[:,0], self.bounds[:,1], size=(n_sample, self.dim))
        return np.squeeze(x)
    
    def sample_airfoil(self, n_sample, method='random'):
        x = self.sample_design_variables(n_sample, method)
        airfoils = self.synthesize(x)
        return airfoils
    
    
class AirfoilGAN(Airfoil):
    
    def __init__(self, latent_dim, noise_dim, model_directory, latent=None, full=False, config_fname='op_conditions.ini'):
        
        self.latent = latent
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        if noise_dim == 0:
            full = False
        self.full = full
        
        if (not full) and (self.latent is None):
            self.dim = self.latent_dim
            self.bounds = np.array([[0., 1.]])
            self.bounds = np.tile(self.bounds, [self.dim, 1])
        else:
            self.dim = self.latent_dim + self.noise_dim
            if self.latent is not None:
                assert len(self.latent) == self.latent_dim
                latent_bounds = np.vstack((latent-0.1, latent+0.1)).T
            else:
                latent_bounds = np.array([0., 1.])
                latent_bounds = np.tile(latent_bounds, [self.latent_dim, 1])
            noise_bounds = np.array([-0.5, 0.5])
            noise_bounds = np.tile(noise_bounds, [self.noise_dim, 1])
            self.bounds = np.vstack((latent_bounds, noise_bounds))
        
        # Expand bounds by 20%
        b = self.bounds
        r = np.max(b, axis=1) - np.min(b, axis=1)
        self.bounds = np.zeros_like(b)
        self.bounds[:,0] = b[:,0] - 0.2*r
        self.bounds[:,1] = b[:,1] + 0.2*r
            
        self.y = None
        self.config_fname = config_fname
        
        self.gan = GAN(self.latent_dim, self.noise_dim, 192, 31, (0., 1.))
        self.gan.restore(model_directory)
        
        n_points = self.gan.X_shape[0]
        x_synth = self.gan.x_fake_test
        x_synth_ = tf.squeeze(x_synth)
        self.x_target = tf.placeholder(tf.float32, shape=[n_points, 2])
        self.e = tf.reduce_mean(tf.reduce_sum(tf.square(x_synth_-self.x_target), axis=1))
        if self.full:
            self.grad_e = tf.concat(tf.gradients(self.e, [self.gan.c, self.gan.z]), axis=1)
        else:
            self.grad_e = tf.gradients(self.e, self.gan.c)
        
    def sample_design_variables(self, n_sample, method='random'):
        if method == 'lhs':
            alpha = lhs(self.dim, samples=n_sample, criterion='cm')
            alpha = alpha * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
        else:
            latent = np.random.rand(n_sample, self.latent_dim)
            if (not self.full) and (self.latent is None):
                alpha = latent
            else:
                noise = np.random.normal(scale=0.5, size=(n_sample, self.noise_dim))
                alpha = np.hstack((latent, noise))
        return np.squeeze(alpha)
    
    def synthesize(self, alpha):
        alpha = np.array(alpha, ndmin=2)
        if (not self.full) and (self.latent is None):
            noise = np.zeros((alpha.shape[0],self.noise_dim))
            airfoils = self.gan.synthesize(alpha, noise)
        else:
            latent = alpha[:, :self.latent_dim]
            noise = alpha[:, self.latent_dim:]
            airfoils = self.gan.synthesize(latent, noise)
        return airfoils
    
    def fit(self, target_airfoil):
        
        def fun(x, target_airfoil, full):
            if not full:
                latent = np.expand_dims(x, axis=0)
                noise = np.zeros((1, self.noise_dim))
            else:
                latent = np.expand_dims(x[:self.latent_dim], axis=0)
                noise = np.expand_dims(x[self.latent_dim:], axis=0)
            f = self.gan.sess.run(self.e, feed_dict={self.gan.c: latent, 
                                                     self.gan.z: noise, 
                                                     self.x_target: target_airfoil})
            return f
        def jac(x, target_airfoil, full):
            if not full:
                latent = np.expand_dims(x, axis=0)
                noise = np.zeros((1, self.noise_dim))
            else:
                latent = np.expand_dims(x[:self.latent_dim], axis=0)
                noise = np.expand_dims(x[self.latent_dim:], axis=0)
            g = self.gan.sess.run(self.grad_e, feed_dict={self.gan.c: latent, 
                                                          self.gan.z: noise, 
                                                          self.x_target: target_airfoil})
            g = np.squeeze(g[0])
            return g
        
        if self.full:
            dim = self.latent_dim + self.noise_dim
        else:
            dim = self.latent_dim
        n_restart = 5*dim
        opt_error = np.inf
        for i in range(n_restart):
            x0 = self.sample_design_variables(1)
            res = minimize(fun, x0, args=(target_airfoil,self.full), jac=jac, method='SLSQP', tol=1e-8)
            airfoil = self.synthesize(res.x)
            error = fun(res.x, target_airfoil, self.full)
            print('{}/{} | error: {:.8f} | success: {} | message: {}'.format(i+1, n_restart, error, res.success, res.message))
            if error < opt_error:
                opt_error = error
                fitted_airfoil = airfoil
                opt_alpha = res.x
        
        return opt_alpha, fitted_airfoil, opt_error
    
    
class AirfoilSVD(Airfoil):
    '''
    References:
    [1] Poole, D. J., Allen, C. B., & Rendall, T. C. (2015). Metric-based mathematical 
        derivation of efficient airfoil design variables. AIAA Journal, 53(5), 1349-1361.
    [2] Poole, D. J., Allen, C. B., & Rendall, T. (2019). Efficient Aero-Structural 
        Wing Optimization Using Compact Aerofoil Decomposition. In AIAA Scitech 2019 Forum (p. 1701).
    '''
    
    def __init__(self, latent_dim, data_path='data/airfoil_interp_uniform.npy', 
                 base_path='initial_airfoil/naca0012_uniform_192.dat', config_fname='op_conditions.ini'):
        
        self.dim = latent_dim
        
        # Read data
        X = np.load(data_path)
        # Split training and test data
        X_train, _ = train_test_plit(X, split=0.8)
        # Select a subset of data
        n = 500
        ind = np.random.choice(X_train.shape[0], n, replace=False)
        X_train = X_train[ind]
        # Make camber line consistent
        y_te = (X_train[:,0,1]+X_train[:,-1,1])/2
        X_train[:,:,1] -= np.expand_dims(y_te, 1)
        
        # SVD for deformation
        X_train = np.transpose(X_train, (0,2,1)).reshape(X_train.shape[0], -1)
        M = X_train.shape[0]
        N = X_train.shape[1]
        print('Computing deformation vectors ...')
        psi = np.zeros((N, M*(M-1)//2)) # N x M(M-1)/2
        for i in range(M):
            for j in range(i+1,M):
                idx = i*(M-1)-i*(i+1)//2+j-1
                psi[:, idx] = np.abs(X_train[i] - X_train[j])
        print('Computing SVD ...')
        u, s, vh = np.linalg.svd(psi, full_matrices=False)
        self.u_truncated = u[:,:self.dim] # N x dim
        self.alpha0 = np.zeros(self.dim)
        self.airfoil0 = np.loadtxt(base_path, delimiter=',')
#        # Plot modes
#        import matplotlib.pyplot as plt
#        for i in range(10):
#            mode = u[:,i].reshape(2,-1).T + self.airfoil0
#            plt.figure()
#            plt.plot(self.airfoil0[:,0], self.airfoil0[:,1], c='b', lw=1)
#            plt.plot(mode[:,0], mode[:,1], c='r', lw=2)
#            for j in range(self.airfoil0.shape[0]):
#                plt.plot([self.airfoil0[j,0], mode[j,0]], [self.airfoil0[j,1], mode[j,1]], c='r', lw=1, alpha=.5)
#            plt.axis('equal')
#            plt.savefig('./svd/mode_{}.svg'.format(i))
#            plt.close()
#        # Plot Laplacian eigenvectors
#        plt.figure()
#        for i in range(5):
#            plt.plot(u[:,i], lw=2)
#        plt.grid()
#        plt.savefig('./svd/singular_vectors.svg')
#        plt.close()
#        # Plot singular values
#        plt.figure()
#        plt.plot(s[:20], 'o-')
#        plt.grid()
#        plt.savefig('./svd/singular_values.svg')
#        plt.close()
#        # Plot singular values in log scale
#        plt.figure()
#        plt.plot(s, '.')
#        plt.yscale("log")
#        plt.grid()
#        plt.savefig('./svd/singular_values_log.svg')
#        plt.close()
#        # Plot retained variance
#        plt.figure()
#        plt.plot(np.cumsum(s)[:20]/np.sum(s), 'o-')
#        plt.grid()
#        plt.savefig('./svd/retained_variance.svg')
#        plt.close()
        
        # Compute latent variables
        self.alpha = np.dot(np.diag(s[:latent_dim]), vh[:latent_dim,:]).T
        self.bounds = np.zeros((latent_dim, 2))
        self.bounds[:,0] = np.min(self.alpha, axis=0)
        self.bounds[:,1] = np.max(self.alpha, axis=0)
            
        self.y = None
        self.config_fname = config_fname
    
    def synthesize(self, alpha):
        alpha = np.array(alpha, ndmin=2).T # dim x n_samples
        airfoils = self.u_truncated @ alpha # N x n_samples
        airfoils = airfoils.reshape(2, -1, alpha.shape[1])
        airfoils = np.transpose(airfoils, [2,1,0]) + self.airfoil0
#        # Adjust trailing head
#        ind = airfoils[:,0,1] < airfoils[:,-1,1]
#        mean = .5*(airfoils[ind,0,1]+airfoils[ind,-1,1])
#        airfoils[ind,0,1] = airfoils[ind,-1,1] = mean
        return np.squeeze(airfoils)
    
    def fit(self, target_airfoil):
        target = target_airfoil - self.airfoil0 # N/2 x 2
        target = target.T.reshape(-1,1) # N x 1
        alpha = np.linalg.pinv(self.u_truncated) @ target # dim x 1
        alpha = np.squeeze(alpha)
        fitted_airfoil = self.synthesize(alpha)
        error = np.mean(np.sum(np.square(fitted_airfoil-target_airfoil), axis=1))
        return alpha, fitted_airfoil, error
    

class AirfoilGeneric(Airfoil):
    '''
    References:
    [1] Kedward, L., Allen, C. B., & Rendall, T. (2020). Towards Generic Modal 
        Design Variables for Aerodynamic Shape Optimisation. In AIAA Scitech 2020 Forum (p. 0543).
    '''
    
    def __init__(self, dim, base_path='initial_airfoil/naca0012_uniform_192.dat', config_fname='op_conditions.ini'):
        
        self.dim = dim
        self.n_points = 192
        N = self.n_points-1
        
        # Differencing matrix
        D1 = -1*np.eye(N)+ np.eye(N)[list(range(1,N))+[0]]
        D2 = D1.T @ D1
        D2[0] = 0
        D3 = D1 @ D2
        D3[:,0] = 0
        D3[:,(N+1)//2] = 0
        D3 = np.tile(D3, [2,1,1])
        
        # SVD for D
        print('Computing SVD ...')
        u, s, vh = np.linalg.svd(D3)
        s = s[:,:-2]
        vh = vh[:,:-2]
        v = np.transpose(vh, [2,1,0]) # n_points x n_points x 2
        self.v_truncated = v[:,-self.dim//2:] # n_points x dim/2 x 2
        self.v_truncated = np.transpose(self.v_truncated, [2,0,1]) # 2 x n_points x dim/2
        self.alpha0 = np.zeros(self.dim)
        self.airfoil0 = np.loadtxt(base_path, delimiter=',')
#        # Plot modes
#        import matplotlib.pyplot as plt
#        for i in range(10):
#            mode = np.vstack((v[:,-i-1], v[:1,-i-1])) + self.airfoil0
#            plt.figure()
#            plt.plot(self.airfoil0[:,0], self.airfoil0[:,1], c='b', lw=1, alpha=.5)
#            plt.plot(mode[:,0], mode[:,1], c='r', lw=2, alpha=.5)
#            for j in range(self.airfoil0.shape[0]):
#                plt.plot([self.airfoil0[j,0], mode[j,0]], [self.airfoil0[j,1], mode[j,1]], c='r', lw=1, alpha=.5)
#            plt.axis('equal')
#            plt.savefig('./generic/mode_{}.svg'.format(i))
#            plt.close()
#        # Apply second difference on Laplacian eigenvectors
#        plt.figure()
#        D2V = D2 @ v[:,-5:,0]
#        for i in range(5):
#            ev = D2V[:,-i-1]
#            plt.plot(ev, lw=2)
#        plt.grid()
#        plt.savefig('./generic/d2_laplacian_eigenvectors.svg')
#        plt.close()
#        # Plot Laplacian eigenvectors
#        plt.figure()
#        for i in range(5):
#            ev = v[:,-i-1,0]
#            plt.plot(ev, lw=2)
#        plt.grid()
#        plt.savefig('./generic/laplacian_eigenvectors.svg')
#        plt.close()
#        # Plot eigenvalues
#        plt.figure()
#        plt.plot(s[0,-20:], 'o-')
#        plt.grid()
#        plt.savefig('./generic/eigenvalues.svg')
#        plt.close()
#        # Plot eigenvalues in log scale
#        plt.figure()
#        plt.plot(s[0], '.')
#        plt.yscale("log")
#        plt.grid()
#        plt.savefig('./generic/eigenvalues_log.svg')
#        plt.close()
        
        # Set bounds
        h_bar = 1./((N+1)//2-1)
        sigma = 100
        epsilon = sigma*h_bar**3
        s = s.T[-dim//2:].flatten()
        self.bounds = np.zeros((dim, 2))
        self.bounds[:,0] = -epsilon/s
        self.bounds[:,1] = epsilon/s
            
        self.y = None
        self.config_fname = config_fname
    
    def synthesize(self, alpha):
        alpha = np.array(alpha, ndmin=2).T # dim x n_samples
        alpha = alpha.reshape(self.dim//2, 2, -1) # dim/2 x 2 x n_samples
        alpha = np.transpose(alpha, [1,0,2]) # 2 x dim/2 x n_samples
        airfoils = self.v_truncated @ alpha # 2 x n_points x n_samples
        airfoils = np.transpose(airfoils, [2,1,0]) # n_samples x n_points x 2
        airfoils = np.concatenate((airfoils, airfoils[:,:1]), axis=1)
        airfoils += self.airfoil0
        return np.squeeze(airfoils)
    
    def fit(self, target_airfoil):
        target = target_airfoil - self.airfoil0 # n_points x 2
        target = target[:-1] # (n_points-1) x 2
        target = target.T.reshape(2,-1,1) # 2 x (n_points-1) x 1
        alpha = np.linalg.pinv(self.v_truncated) @ target # 2 x dim/2 x 1
        alpha = np.squeeze(alpha) # 2 x dim/2
        alpha = alpha.T.flatten()
        fitted_airfoil = self.synthesize(alpha)
        error = np.mean(np.sum(np.square(fitted_airfoil-target_airfoil), axis=1))
        return alpha, fitted_airfoil, error
    
    
class AirfoilFFD(Airfoil):
    '''
    Reference:
        Masters, D. A., Taylor, N. J., Rendall, T. C. S., Allen, C. B., & Poole, D. J. (2017). 
        Geometric comparison of aerofoil shape parameterization methods. AIAA Journal, 1575-1589.
    '''
    
    def __init__(self, m=4, n=3, initial_path='initial_airfoil/naca0012.dat', config_fname='op_conditions.ini'):
        
        # Airfoil parameters
        self.m = m
        self.n = n
        
        # NACA 0012 as the initial airfoil
        try:
            self.airfoil0 = np.loadtxt(initial_path, skiprows=1)
        except:
            self.airfoil0 = np.loadtxt(initial_path, delimiter=',')
        x_min = np.min(self.airfoil0[:,0])
        x_max = np.max(self.airfoil0[:,0])
        z_min = np.min(self.airfoil0[:,1])
        z_max = np.max(self.airfoil0[:,1])
        Px = np.linspace(x_min, x_max, self.m, endpoint=True)
        Py = np.linspace(z_min, z_max, self.n, endpoint=True)
        x, y = np.meshgrid(Px, Py)
        P0 = np.stack((x, y), axis=-1)
        self.Px = P0[:,:,0]
        self.alpha0 = P0[:,:,1].flatten()
        
        self.dim = len(self.alpha0)
        self.bounds = np.zeros((self.dim, 2))
        perturb = 0.2
        self.bounds[:,0] = self.alpha0 - perturb
        self.bounds[:,1] = self.alpha0 + perturb
            
        self.y = None
        self.config_fname = config_fname
    
    def synthesize(self, alpha):
        alpha = np.array(alpha, ndmin=2)
        airfoils = np.apply_along_axis(lambda x: synthesize_ffd(x, self.airfoil0, self.m, self.n, self.Px), 1, alpha)
        return np.squeeze(airfoils)
    
    def fit(self, target_airfoil):
        
        def fun(x, target_airfoil):
            airfoil = synthesize_ffd(x, self.airfoil0, self.m, self.n, self.Px)
            error = np.mean(np.sum(np.square(airfoil-target_airfoil), axis=1))
            return error
        def jac(x, target_airfoil):
            fun_ = lambda x: fun(x, target_airfoil)
            return grad(fun_)(x)
        
        n_restart = 5*self.dim
        opt_error = np.inf
        for i in range(n_restart):
            x0 = self.sample_design_variables(1)
            res = minimize(fun, x0, args=(target_airfoil,), jac=jac, method='SLSQP', tol=1e-8)
            airfoil = self.synthesize(res.x)
            error = fun(res.x, target_airfoil)
            print('{}/{} | error: {:.8f} | success: {} | message: {}'.format(i+1, n_restart, error, res.success, res.message))
            if error < opt_error:
                opt_error = error
                fitted_airfoil = airfoil
                opt_alpha = res.x
        
        return opt_alpha, fitted_airfoil, opt_error
    
    
    

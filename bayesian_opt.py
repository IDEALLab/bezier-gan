from __future__ import division
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, bisect
import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from pyDOE import lhs
from gp import GPR


def normalize(y, return_mean_std=False):
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y-y_mean)/y_std
    if return_mean_std:
        return y, y_mean, y_std
    return y

def inv_normalize(y, y_mean, y_std):
    return y*y_std + y_mean

def proba_of_improvement(samples, gp_model, f_best):
    samples = np.array(samples).reshape(-1, gp_model.X_train_.shape[1])
    mu, sigma = gp_model.predict(samples, return_std=True)
    mu = mu.reshape(-1,1)
    sigma = sigma.reshape(-1,1)
    PI = 1 - norm.cdf(f_best, loc=mu, scale=sigma)
    return np.squeeze(PI)

def expected_improvement(samples, gp_model, f_best):
    samples = np.array(samples).reshape(-1, gp_model.X_train_.shape[1])
    mu, sigma = gp_model.predict(samples, return_std=True)
    mu = mu.reshape(-1,1)
    sigma = sigma.reshape(-1,1)
    with np.errstate(divide='ignore'):
        Z = (mu - f_best)/sigma
        EI = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma==0.0] = 0.0
    return np.squeeze(EI)

def upper_confidence_bound(samples, gp_model, beta):
    samples = np.array(samples).reshape(-1, gp_model.X_train_.shape[1])
    mu, sigma = gp_model.predict(samples, return_std=True)
    mu = mu.reshape(-1,1)
    sigma = sigma.reshape(-1,1)
    UCB = mu + beta * sigma
    return np.squeeze(UCB)

def lower_confidence_bound(samples, gp_model, beta):
    samples = np.array(samples).reshape(-1, gp_model.X_train_.shape[1])
    mu, sigma = gp_model.predict(samples, return_std=True)
    mu = mu.reshape(-1,1)
    sigma = sigma.reshape(-1,1)
    LCB = mu - beta * sigma
    return np.squeeze(LCB)

def regularized_ei_quadratic(samples, gp_model, f_best, center, w):
    samples = np.array(samples).reshape(-1, gp_model.X_train_.shape[1])
    mu, sigma = gp_model.predict(samples, return_std=True)
    mu = mu.reshape(-1,1)
    sigma = sigma.reshape(-1,1)
    epsilon = np.diag(np.matmul(np.matmul(samples-center, np.diag(w**(-2))), (samples-center).T)).reshape(-1,1)
    f_tilde = f_best * (1. + np.sign(f_best)*epsilon)
    with np.errstate(divide='ignore'):
        Z = (mu - f_tilde)/sigma
        EIQ = (mu - f_tilde) * norm.cdf(Z) + sigma * norm.pdf(Z)
        EIQ[sigma==0.0] = 0.0
    return np.squeeze(EIQ)

def regularized_ei_hinge_quadratic(samples, gp_model, f_best, center, R, beta):
    samples = np.array(samples).reshape(-1, gp_model.X_train_.shape[1])
    mu, sigma = gp_model.predict(samples, return_std=True)
    mu = mu.reshape(-1,1)
    sigma = sigma.reshape(-1,1)
    dists = np.linalg.norm(samples-center, axis=1, keepdims=True)
    epsilon = (dists-R)/beta/R
    epsilon[dists < R] = 0.0
    f_tilde = f_best * (1 + np.sign(f_best)*epsilon)
    with np.errstate(divide='ignore'):
        Z = (mu - f_tilde)/sigma
        EIQ = (mu - f_tilde) * norm.cdf(Z) + sigma * norm.pdf(Z)
        EIQ[sigma==0.0] = 0.0
    return np.squeeze(EIQ)

def var_constrained_pi(samples, gp_model, f_best, tau):
    samples = np.array(samples).reshape(-1, gp_model.X_train_.shape[1])
    mu, sigma = gp_model.predict(samples, return_std=True)
    mu = mu.reshape(-1,1)
    sigma = sigma.reshape(-1,1)
    PI = 1 - norm.cdf(f_best, loc=mu, scale=sigma)
    PI[sigma > (tau*gp_model.kernel_.diag(samples).reshape(-1,1))**.5] = 0.0
    return np.squeeze(PI)

def var_constrained_ei(samples, gp_model, f_best, tau):
    samples = np.array(samples).reshape(-1, gp_model.X_train_.shape[1])
    mu, sigma = gp_model.predict(samples, return_std=True)
    mu = mu.reshape(-1,1)
    sigma = sigma.reshape(-1,1)
    with np.errstate(divide='ignore'):
        Z = (mu - f_best)/sigma
        EI = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma==0.0] = 0.0
    EI[sigma > (tau*gp_model.kernel_.diag(samples).reshape(-1,1))**.5] = 0.0
    return np.squeeze(EI)

def compute_tau(f_best, gp_model, xi=0.01, kappa=0.1):
    delta = 0.01
    sigma_plus = (xi+delta)/norm.ppf(1-kappa)
    _ = np.zeros((1, gp_model.X_train_.shape[1]))
    k0 = np.asscalar(gp_model.kernel_(_, _))
    def func(x):
        mu_tau = 0.#xi - np.sqrt(x*k0)*norm.ppf(1-kappa)
        u_tau = (mu_tau-f_best)/np.sqrt(x*k0)
        EI_tau = np.sqrt(x*k0) * (u_tau*norm.cdf(u_tau) + norm.pdf(u_tau))
        u_plus = -delta/sigma_plus
        EI_plus = sigma_plus * (u_plus*norm.cdf(u_plus) + norm.pdf(u_plus))
        return EI_tau - EI_plus
#    import matplotlib.pyplot as plt
#    xx = np.linspace(0.1, 1., 100)
#    plt.plot(xx, func(xx))
    try:
        tau = bisect(func, 0.01, 1.)
        tau = np.clip(tau, 0.0001, 0.99)
    except ValueError:
        tau = 0.99
    return tau

def constraint_proba(samples, gpc_model):
    samples = np.array(samples).reshape(-1, gpc_model.base_estimator_.X_train_.shape[1])
    pr = gpc_model.predict_proba(samples)[:,1]
    return np.squeeze(pr)

def constraint_weighted_acquisition(samples, acquisition_func, constraint_proba_func, delta=0.5):
    afv = acquisition_func(samples)
    pr = constraint_proba_func(samples)
    afv *= pr
    afv[pr<1-delta] = 0.0
    return afv

def grad_ei(samples, gp_model, f_best):
    samples = np.array(samples).reshape(-1, gp_model.X_train_.shape[1])
    mu, sigma = gp_model.predict(samples, return_std=True)
    mu = mu.reshape(-1,1)
    sigma = sigma.reshape(-1,1)
    u = (mu - f_best) / sigma
    dmu_dx, dsigma_dx = gp_model.grad_predict(samples, compute_std=True)
    du_dx = (dmu_dx - u*dsigma_dx)/sigma
    dEI_dx = (u*norm.cdf(u)+norm.pdf(u))*dsigma_dx + sigma*norm.cdf(u)*du_dx
    return dEI_dx

def generate_candidates(d, n_candidates, bounds=None, gaussian=None, ball=None):
    
    if bounds is not None:
        bounds = np.array(bounds, ndmin=2)
        candidates = np.random.uniform(bounds[:,0], bounds[:,1], size=(n_candidates, d))
        
    elif gaussian is not None:
        mean = np.array(gaussian[0], ndmin=1)
        cov = np.array(gaussian[1], ndmin=2)
        candidates = np.random.multivariate_normal(mean, cov, size=n_candidates)
        
    elif ball is not None:
        def sample_sphere(center, radius, num):
            count = 0
            samples = []
            while count < num:
                sample = np.random.uniform(-radius, radius, d)
                if np.linalg.norm(sample) <= radius:
                    samples.append(sample + center)
                    count += 1
            samples = np.array(samples)
            return samples
        center = ball[0]
        radius = ball[1]
        candidates = sample_sphere(center, radius, n_candidates)
        
    else:
        candidates = np.random.rand(n_candidates, d)
        
    return candidates

def sample_next_point(d, acquisition_func, candidates=None, bounds=None, strict_bounds=False, gaussian=None, ball=None, 
                      n_candidates=1000, n_restarts=1, random_search=False, return_opt_f=False):
    
    opt_x = None
    f = lambda x: -acquisition_func(x)
    
    if candidates is None:
        candidates = generate_candidates(d, n_candidates, bounds, gaussian, ball)
    
    # Random search
    if random_search:
        afv = np.squeeze(f(candidates))
        opt_x = candidates[np.argmin(afv)]
    
    # L-BFGS-B
    else:
        f_candidates = f(candidates).flatten()
        x0s = candidates[np.argsort(f_candidates)[:n_restarts]]
        opt_f = np.inf
        if strict_bounds and bounds is not None:
            bs = np.array(bounds, ndmin=2)
        else:
            bs = None
        for x0 in x0s:
            res = minimize(fun=f,
                           x0=x0,
                           bounds=bs,
                           method='L-BFGS-B')
            if res.fun < opt_f:
                opt_f = res.fun
                opt_x = res.x
              
    if return_opt_f:
        return opt_x, -opt_f
    
    return opt_x

def bo_c(func, n_eval, n_init_eval, n_candidates, bounds, alpha=1e-4, save_dir=None):
    
#    kernel = gp.kernels.Matern()
    kernel = gp.kernels.ConstantKernel(1.0, (1., 1.)) * gp.kernels.RBF(1.0, (1e-5, 1e5))
    gp_model = GPR(kernel=kernel, alpha=alpha, n_restarts_optimizer=100, normalize_y=False)
    gpc_model = GPC(kernel=kernel, n_restarts_optimizer=100)
    
    dim = func.dim
    
    # Initial evaluations
    xs = lhs(dim, samples=n_init_eval, criterion='cm')
    xs = xs * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
    ys = func(xs)
    vs = func.is_feasible(xs)
    
    opt_idx = np.argmax(ys[vs])
    opt_x = xs[vs][opt_idx]
    opt_y = ys[vs][opt_idx]
    
    opt_ys = [opt_y]
    
    for i in range(n_init_eval, n_eval):
        
        ys_normalized = normalize(ys[vs])
        gp_model.fit(xs[vs], ys_normalized)
        f_prime = ys_normalized[opt_idx]
        acquisition_func = lambda x: expected_improvement(x, gp_model, f_prime)
        
        if np.any(vs) and np.any(np.logical_not(vs)):
            gpc_model.fit(xs, vs)
            constraint_proba_func = lambda x: constraint_proba(x, gpc_model)
            constraint_weighted_acquisition_func = lambda x: constraint_weighted_acquisition(x, acquisition_func, constraint_proba_func)
        else:
            constraint_weighted_acquisition_func = acquisition_func
        
        # Decide point to evaluate next
        n_candidates = 1000*dim
        x = sample_next_point(dim, constraint_weighted_acquisition_func, bounds=bounds, strict_bounds=True, n_candidates=n_candidates)
        
        y = func(x)
        v = func.is_feasible(x)
        xs = np.append(xs, np.array(x, ndmin=2), axis=0)
        ys = np.append(ys, y)
        vs = np.append(vs, v)
        
        if v and y > opt_y:
            opt_idx = sum(vs) - 1
            
        opt_x = xs[vs][opt_idx]
        opt_y = ys[vs][opt_idx]
        opt_ys.append(opt_y) # Best performance so far
        print('{}: x {} y {} v {} Best-so-far {}'.format(i+1, x, y, v, opt_y))
        
    return opt_x, opt_ys

def optimize(n_eval, n_init_eval, func):
    
    dim = func.dim
    n_candidates = 1000*dim
    bounds = func.bounds
    
    opt_x, opt_ys = bo_c(func, n_eval, n_init_eval, n_candidates, bounds)
    print('Optimal: x {} CL/CD {}'.format(opt_x, opt_ys[-1]))
        
    opt_airfoil = func.synthesize(opt_x)
    opt_ys = np.hstack((np.nan*np.ones(n_init_eval), opt_ys))
        
    return opt_x, opt_airfoil, opt_ys


"""
Fitting test.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import tensorflow as tf

import functions
from utils import create_dir


def plot(fitted_airfoil, target_airfoil, max_dist_idx=None, P=None, W=None):
    from simulation import evaluate
    perf, CL, CD = evaluate(fitted_airfoil, return_CL_CD=True)
    print('Approximated:', CL, CD)
    perf, CL, CD = evaluate(target_airfoil, return_CL_CD=True)
    print('Target:', CL, CD)
    plt.figure()
    plt.plot(fitted_airfoil[:,0], fitted_airfoil[:,1], 'ro-', alpha=.7, label='approximated')
    plt.plot(target_airfoil[:,0], target_airfoil[:,1], 'bo-', alpha=.7, label='target')
    plt.legend(frameon=False)
    if max_dist_idx is not None:
        plt.plot(fitted_airfoil[max_dist_idx, 0], fitted_airfoil[max_dist_idx,1], 'rx', ms=20)
        plt.plot(target_airfoil[max_dist_idx, 0], target_airfoil[max_dist_idx,1], 'bx', ms=20)
    if P is not None:
        if W is None:
            W = 0
        plt.plot(P[:,0], P[:,1], 'r-', alpha=.7)
        plt.scatter(P[:,0], P[:,1], marker='s', c='r', s=10+1000*W, alpha=.7)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    
#    ''' BezierGAN '''
#    database = np.load('data/airfoil_interp.npy')
#    latent_dim = 20
#    noise_dim = 10
#    model_directory = './beziergan/trained_gan/{}_{}/0'.format(latent_dim, noise_dim)
#    # Test code
#    target_airfoil = database[np.random.choice(database.shape[0])]
#    tf.keras.backend.clear_session()
#    func = functions.AirfoilGAN(latent_dim, noise_dim, model_directory, full=False)
#    alpha, fitted_airfoil, error = func.fit(target_airfoil)
#    plot(fitted_airfoil, target_airfoil)
#    print(error)
    
#    ''' SVD '''
#    database = np.load('data/airfoil_interp_uniform.npy')
#    latent_dim = 14
#    # Test code
#    target_airfoil = database[np.random.choice(database.shape[0])]
#    func = functions.AirfoilSVD(latent_dim)
#    alpha, fitted_airfoil, error = func.fit(target_airfoil)
#    plot(fitted_airfoil, target_airfoil)
#    print(error)
    
#    ''' GMDV '''
#    database = np.load('data/airfoil_interp_uniform.npy')
#    dim = 14
#    # Test code
#    target_airfoil = database[np.random.choice(database.shape[0])]
#    func = functions.AirfoilGeneric(dim)
#    alpha, fitted_airfoil, error = func.fit(target_airfoil)
#    plot(fitted_airfoil, target_airfoil)
#    print(error)
        
#    ''' FFD '''
#    database = np.load('data/airfoil_interp_uniform.npy')
#    dim = 6
#    # Test code
#    target_airfoil = database[np.random.choice(database.shape[0])]
#    func = functions.AirfoilFFD(m=dim//3, n=3, initial_path='initial_airfoil/naca0012_uniform_192.dat')
#    alpha, fitted_airfoil, error = func.fit(target_airfoil)
#    P = np.stack((func.Px, alpha.reshape(3,dim//3)), axis=-1).reshape(-1,2)
#    plot(fitted_airfoil, target_airfoil, P=P)
#    print(error)
        
    
    results_dir = 'results_fit'
    create_dir(results_dir)
    
    ''' BezierGAN '''
    noise_dim = 10
    list_dims_gan = [2, 6, 8, 10, 14, 20]
    database = np.load('data/airfoil_interp.npy')
    gan_mean_errors = []
    for latent_dim in list_dims_gan:
        model_directory = './beziergan/trained_gan/{}_{}/0'.format(latent_dim, noise_dim)
        fname = '{}/fitting_errors_gan_{}_{}.npy'.format(results_dir, latent_dim, noise_dim)
        if os.path.exists(fname):
            errors = np.load(fname)
        else:
            tf.keras.backend.clear_session()
            func = functions.AirfoilGAN(latent_dim, noise_dim, model_directory, full=False)
            errors = []
            for i, target_airfoil in enumerate(database):
                alpha, fitted_airfoil, error = func.fit(target_airfoil)
                errors.append(error)
                print('BezierGAN {} {}: {}/{}: {}'.format(latent_dim, noise_dim, i+1, database.shape[0], error))
            np.save(fname, errors)
        mean_error = np.mean(errors)
        gan_mean_errors.append(mean_error)
        
    list_dims_gan_full = [2, 6, 8, 10, 14]
    database = np.load('data/airfoil_interp.npy')
    gan_full_mean_errors = []
    for latent_dim in list_dims_gan_full:
        model_directory = './beziergan/trained_gan/{}_{}/0'.format(latent_dim, noise_dim)
        fname = '{}/fitting_errors_gan_{}_{}_full.npy'.format(results_dir, latent_dim, noise_dim)
        if os.path.exists(fname):
            errors = np.load(fname)
        else:
            tf.keras.backend.clear_session()
            func = functions.AirfoilGAN(latent_dim, noise_dim, model_directory, full=True)
            errors = []
            for i, target_airfoil in enumerate(database):
                alpha, fitted_airfoil, error = func.fit(target_airfoil)
                errors.append(error)
                print('BezierGAN full {} {}: {}/{}: {}'.format(latent_dim, noise_dim, i+1, database.shape[0], error))
            np.save(fname, errors)
        mean_error = np.mean(errors)
        gan_full_mean_errors.append(mean_error)
        
    ''' SVD '''
    list_dims_svd = [2, 6, 8, 9, 10, 14, 20]
    database = np.load('data/airfoil_interp_uniform.npy')
    svd_mean_errors = []
    for latent_dim in list_dims_svd:
        fname = '{}/fitting_errors_svd_{}.npy'.format(results_dir, latent_dim)
        if os.path.exists(fname):
            errors = np.load(fname)
        else:
            func = functions.AirfoilSVD(latent_dim)
            errors = []
            for i, target_airfoil in enumerate(database):
                alpha, fitted_airfoil, error = func.fit(target_airfoil)
                errors.append(error)
                print('SVD {}: {}/{}: {}'.format(latent_dim, i+1, database.shape[0], error))
            np.save(fname, errors)
        mean_error = np.mean(errors)
        svd_mean_errors.append(mean_error)
        
    ''' GMDV '''
    list_dims_generic = [2, 6, 8, 10, 14, 20]
    database = np.load('data/airfoil_interp_uniform.npy')
    generic_mean_errors = []
    for dim in list_dims_generic:
        fname = '{}/fitting_errors_generic_{}.npy'.format(results_dir, dim)
        if os.path.exists(fname):
            errors = np.load(fname)
        else:
            func = functions.AirfoilGeneric(dim)
            errors = []
            for i, target_airfoil in enumerate(database):
                alpha, fitted_airfoil, error = func.fit(target_airfoil)
                errors.append(error)
                print('Generic {}: {}/{}: {}'.format(dim, i+1, database.shape[0], error))
            np.save(fname, errors)
        mean_error = np.mean(errors)
        generic_mean_errors.append(mean_error)
        
    ''' FFD '''
    list_dims_ffd = [6, 9, 12, 15, 18]
    database = np.load('data/airfoil_interp_uniform.npy')
    ffd_mean_errors = []
    for dim in list_dims_ffd:
        fname = '{}/fitting_errors_ffd_{}.npy'.format(results_dir, dim)
        if os.path.exists(fname):
            errors = np.load(fname)
        else:
            func = functions.AirfoilFFD(m=dim//3, n=3, initial_path='initial_airfoil/naca0012_uniform_192.dat')
            errors = []
            for i, target_airfoil in enumerate(database):
                alpha, fitted_airfoil, error = func.fit(target_airfoil)
                errors.append(error)
                print('FFD {}: {}/{}: {}'.format(dim, i+1, database.shape[0], error))
            np.save(fname, errors)
        mean_error = np.mean(errors)
        ffd_mean_errors.append(mean_error)
        
    # Plot
    plt.figure(figsize=(14,5))
    
    plt.subplot(121)
    plt.plot(list_dims_gan, gan_mean_errors, 'o-.', label=r'B$\acute{e}$zier-GAN (fix noise)', alpha=.7)
    plt.plot(list_dims_svd, svd_mean_errors, 's--', label='SVD', alpha=.7)
    plt.plot(list_dims_generic, generic_mean_errors, '^:', label='GMDV', alpha=.7)
#    plt.plot(list_dims_ffd, ffd_mean_errors, 'D-.', label='FFD', alpha=.7)
    plt.ylim([0, 0.001])
    # plt.ylim([1e-6, 1e-3])
    # plt.yscale('log')
    plt.legend(frameon=False)
    plt.xlabel('Number of variables')
    plt.ylabel('Mean square error')
    
    plt.subplot(122)
    plt.plot(list_dims_gan, gan_mean_errors, 'o-.', label=r'B$\acute{e}$zier-GAN (fix noise)', alpha=.7)
    plt.plot(list_dims_gan_full, gan_full_mean_errors, 's-', label=r'B$\acute{e}$zier-GAN (optimize noise)', alpha=.7)
    plt.ylim([0, 0.001])
    # plt.ylim([1e-6, 1e-3])
    # plt.yscale('log')
    plt.legend(frameon=False)
    plt.xlabel('Latent dimension')
    
    plt.tight_layout()
    plt.savefig(results_dir+'/fitting_errors.svg')
    plt.savefig(results_dir+'/fitting_errors.pdf')
    plt.close()
        
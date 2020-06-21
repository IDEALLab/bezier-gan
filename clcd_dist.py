"""
Plot distribution of CL and CD.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import numpy as np
from scipy.stats import kde
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import tensorflow as tf

import functions
from simulation import evaluate, detect_intersect
from utils import create_dir


def plot_density(clcd, pos=None):
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
#    k = kde.gaussian_kde(clcd.T)
#    nbins = 20
#    xlim = (np.min(clcd[:,0]), np.max(clcd[:,0]))
#    ylim = (np.min(clcd[:,1]), np.max(clcd[:,1]))
#    xi, yi = np.mgrid[xlim[0]:xlim[1]:nbins*1j, ylim[0]:ylim[1]:nbins*1j]
#    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='Blues')
#    if pos=='last':
#        plt.colorbar(label='Density')
#    else:
#        plt.colorbar()
    plt.scatter(clcd[:,0], clcd[:,1], marker='s', s=5, alpha=.3, edgecolors='none')
    plt.xlabel(r'$C_L$')
    if pos=='first':
        plt.ylabel(r'$C_D$')


if __name__ == "__main__":
    
    results_dir = 'results_clcd'
    create_dir(results_dir)
    
    database = np.load('data/airfoil_interp.npy')
    N = database.shape[0]
    
    ''' Data '''
    fname = '{}/clcd_data.npy'.format(results_dir)
    if os.path.exists(fname):
        clcd_data = np.load(fname)
    else:
        list_cl = []
        list_cd = []
        for i, airfoil in enumerate(database):
            if detect_intersect(airfoil):
                te = (airfoil[0]+airfoil[-1])/2
                airfoil[0] = airfoil[-1] = te
            _, cl, cd = evaluate(airfoil, return_CL_CD=True)
            if np.isnan(cl) or np.isnan(cd):
                cl = 0
                cd = 0
            print('{}/{}:  CL={:.6f}  CD={:.6f}'.format(i, N, cl, cd))
            list_cl.append(cl)
            list_cd.append(cd)
        clcd_data = np.vstack((list_cl, list_cd)).T
        np.save(fname, clcd_data)
        
    ''' BezierGAN '''
    latent_dim = 8
    noise_dim = 10
    fname = '{}/clcd_gan_{}_{}.npy'.format(results_dir, latent_dim, noise_dim)
    if os.path.exists(fname):
        clcd_gan = np.load(fname)
    else:
        tf.keras.backend.clear_session()
        model_directory = './beziergan/trained_gan/{}_{}/0'.format(latent_dim, noise_dim)
        func = functions.AirfoilGAN(latent_dim, noise_dim, model_directory, full=True)
        airfoils = func.sample_airfoil(N)
        list_cl = []
        list_cd = []
        for i, airfoil in enumerate(airfoils):
            if detect_intersect(airfoil):
                te = (airfoil[0]+airfoil[-1])/2
                airfoil[0] = airfoil[-1] = te
            _, cl, cd = evaluate(airfoil, return_CL_CD=True)
            if np.isnan(cl) or np.isnan(cd):
                cl = 0
                cd = 0
            print('{}/{}:  CL={:.6f}  CD={:.6f}'.format(i, N, cl, cd))
            list_cl.append(cl)
            list_cd.append(cd)
        clcd_gan = np.vstack((list_cl, list_cd)).T
        np.save(fname, clcd_gan)
        
    ''' SVD '''
    latent_dim = 9
    fname = '{}/clcd_svd_{}.npy'.format(results_dir, latent_dim)
    if os.path.exists(fname):
        clcd_svd = np.load(fname)
    else:
        func = functions.AirfoilSVD(latent_dim)
        airfoils = func.sample_airfoil(N)
        list_cl = []
        list_cd = []
        for i, airfoil in enumerate(airfoils):
            if detect_intersect(airfoil):
                te = (airfoil[0]+airfoil[-1])/2
                airfoil[0] = airfoil[-1] = te
            _, cl, cd = evaluate(airfoil, return_CL_CD=True)
            if np.isnan(cl) or np.isnan(cd):
                cl = 0
                cd = 0
            print('{}/{}:  CL={:.6f}  CD={:.6f}'.format(i, N, cl, cd))
            list_cl.append(cl)
            list_cd.append(cd)
        clcd_svd = np.vstack((list_cl, list_cd)).T
        np.save(fname, clcd_svd)
        
    ''' GMDV '''
    dim = 8
    fname = '{}/clcd_generic_{}.npy'.format(results_dir, dim)
    if os.path.exists(fname):
        clcd_generic = np.load(fname)
    else:
        func = functions.AirfoilGeneric(dim)
        airfoils = func.sample_airfoil(N)
        list_cl = []
        list_cd = []
        for i, airfoil in enumerate(airfoils):
            if detect_intersect(airfoil):
                te = (airfoil[0]+airfoil[-1])/2
                airfoil[0] = airfoil[-1] = te
            _, cl, cd = evaluate(airfoil, return_CL_CD=True)
            if np.isnan(cl) or np.isnan(cd):
                cl = 0
                cd = 0
            print('{}/{}:  CL={:.6f}  CD={:.6f}'.format(i, N, cl, cd))
            list_cl.append(cl)
            list_cd.append(cd)
        clcd_generic = np.vstack((list_cl, list_cd)).T
        np.save(fname, clcd_generic)
        
    ''' FFD '''
    dim = 12
    fname = '{}/clcd_ffd_{}.npy'.format(results_dir, dim)
    if os.path.exists(fname):
        clcd_ffd = np.load(fname)
    else:
        func = functions.AirfoilFFD(m=dim//3, n=3, initial_path='initial_airfoil/naca0012_uniform_192.dat')
        airfoils = func.sample_airfoil(N)
        list_cl = []
        list_cd = []
        for i, airfoil in enumerate(airfoils):
            if detect_intersect(airfoil):
                te = (airfoil[0]+airfoil[-1])/2
                airfoil[0] = airfoil[-1] = te
            _, cl, cd = evaluate(airfoil, return_CL_CD=True)
            if np.isnan(cl) or np.isnan(cd):
                cl = 0
                cd = 0
            print('{}/{}:  CL={:.6f}  CD={:.6f}'.format(i, N, cl, cd))
            list_cl.append(cl)
            list_cd.append(cd)
        clcd_ffd = np.vstack((list_cl, list_cd)).T
        np.save(fname, clcd_ffd)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plot_density(clcd_data, 'first')
    plt.title('UIUC database')
    plt.subplot(232)
    plot_density(clcd_gan)
    plt.title(r'B$\acute{e}$zier-GAN')
    plt.subplot(233)
    plot_density(clcd_svd)
    plt.title('SVD')
    plt.subplot(235)
    plot_density(clcd_generic, 'first')
    plt.title('GMDV')
    plt.subplot(236)
    plot_density(clcd_ffd)
    plt.title('FFD')
    plt.tight_layout()
    plt.savefig(results_dir+'/clcd.svg')
    plt.savefig(results_dir+'/clcd.pdf')
    plt.close()
        
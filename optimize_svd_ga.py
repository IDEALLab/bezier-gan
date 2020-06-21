"""
Optimize the airfoil shape directly using genetic algorithm, 
constrained on the running time

Author(s): Wei Chen (wchen459@umd.edu)

Reference(s):
    Viswanath, A., J. Forrester, A. I., Keane, A. J. (2011). Dimension Reduction for Aerodynamic Design Optimization.
    AIAA Journal, 49(6), 1256-1266.
    Grey, Z. J., Constantine, P. G. (2018). Active subspaces of airfoil shape parameterizations.
    AIAA Journal, 56(5), 2003-2017.
"""

from __future__ import division
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from functions import AirfoilSVD
from genetic_alg import optimize
from utils import mean_err, create_dir
    

if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Optimize')
    parser.add_argument('dimension', type=int, default=3, help='design variable dimension')
    parser.add_argument('--n_runs', type=int, default=10, help='number of runs')
    parser.add_argument('--n_eval', type=int, default=1000, help='number of evaluations per run')
    args = parser.parse_args()
    
    n_runs = args.n_runs
    n_eval = args.n_eval
    
    dim = args.dimension
    func = AirfoilSVD(latent_dim=dim)
    
    perturb_type = 'absolute'
    perturb = 0.1
    
    opt_x_runs = []
    opt_airfoil_runs = []
    opt_perfs_runs = []
    time_runs = []
    
    for i in range(n_runs):
                
        print('')
        print('######################################################')
        print('# Method: SVD-GA')
        print('# Run: {}/{}'.format(i+1, n_runs))
        print('######################################################')
              
        start_time = time.time()
        opt_x, opt_airfoil, opt_perfs = optimize(func, perturb_type, perturb, n_eval)
        end_time = time.time()
        opt_x_runs.append(opt_x)
        opt_airfoil_runs.append(opt_airfoil)
        opt_perfs_runs.append(opt_perfs)
        time_runs.append(end_time-start_time)
    
    opt_x_runs = np.array(opt_x_runs)
    opt_airfoil_runs = np.array(opt_airfoil_runs)
    opt_perfs_runs = np.array(opt_perfs_runs)
    save_dir = 'results_opt/svd_ga_{}'.format(dim)
    create_dir(save_dir)
    np.save('{}/opt_solution.npy'.format(save_dir), opt_x_runs)
    np.save('{}/opt_airfoil.npy'.format(save_dir), opt_airfoil_runs)
    np.save('{}/opt_history.npy'.format(save_dir), opt_perfs_runs)
    
    # Plot optimization history
    mean_perfs_runs = np.mean(opt_perfs_runs, axis=0)
    plt.figure()
    plt.plot(np.arange(n_eval+1, dtype=int), opt_perfs)
    plt.title('Optimization History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Optimal CL/CD')
#    plt.xticks(np.linspace(0, n_eval+1, 5, dtype=int))
    plt.savefig('{}/opt_history.svg'.format(save_dir))
    plt.close()
    
    # Plot the optimal airfoil
    mean_time_runs, err_time_runs = mean_err(time_runs)
    mean_final_perf_runs, err_final_perf_runs = mean_err(opt_perfs_runs[:,-1])
    plt.figure()
    for opt_airfoil in opt_airfoil_runs:
        plt.plot(opt_airfoil[:,0], opt_airfoil[:,1], '-', c='k', alpha=1.0/n_runs)
    plt.title('CL/CD: %.2f+/-%.2f  time: %.2f+/-%.2f min' % (mean_final_perf_runs, err_final_perf_runs, 
                                                             mean_time_runs/60, err_time_runs/60))
    plt.axis('equal')
    plt.savefig('{}/opt_airfoil.svg'.format(save_dir))
    plt.close()

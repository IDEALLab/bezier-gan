
"""
Optimize the airfoil shape in the latent space using Bayesian optimization, 
constrained on the running time

Author(s): Wei Chen (wchen459@umd.edu)
"""

from __future__ import division
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from bayesian_opt import optimize
from functions import AirfoilGeneric
from utils import mean_err, create_dir


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Optimize')
    parser.add_argument('dimension', type=int, default=3, help='design variable dimension')
    parser.add_argument('--n_runs', type=int, default=10, help='number of runs')
    parser.add_argument('--n_eval', type=int, default=1000, help='number of evaluations per run')
    args = parser.parse_args()
    
    dim = args.dimension
    n_runs = args.n_runs
    n_eval = args.n_eval
    n_init_eval = 10
    func = AirfoilGeneric(dim=dim)
    
    opt_x_runs = []
    opt_airfoil_runs = []
    opt_perfs_runs = []
    time_runs = []
    
    for i in range(n_runs):
                
        print('')
        print('######################################################')
        print('# Method: Generic-EGO {}'.format(dim))
        print('# Run: {}/{}'.format(i+1, n_runs))
        print('######################################################')
              
        successful = False
        while not successful:
            try:
                start_time = time.time()
                opt_x, opt_airfoil, opt_perfs = optimize(n_eval, n_init_eval, func)
                end_time = time.time()
                opt_x_runs.append(opt_x)
                opt_airfoil_runs.append(opt_airfoil)
                opt_perfs = opt_perfs.tolist()
                opt_perfs_runs.append(opt_perfs)
                time_runs.append(end_time-start_time)
                successful = True
            except Exception as e:
                print(e)
            
    opt_x_runs = np.array(opt_x_runs)
    opt_airfoil_runs = np.array(opt_airfoil_runs)
    opt_perfs_runs = np.array(opt_perfs_runs)
    save_dir = 'results_opt/generic_bo_{}'.format(dim)
    create_dir(save_dir)
    np.save('{}/opt_solution.npy'.format(save_dir), opt_x_runs)
    np.save('{}/opt_airfoil.npy'.format(save_dir), opt_airfoil_runs)
    np.save('{}/opt_history.npy'.format(save_dir), opt_perfs_runs)
    
    # Plot optimization history
    mean_perfs_runs = np.mean(opt_perfs_runs, axis=0)
    plt.figure()
    plt.plot(np.arange(n_eval+1, dtype=int), mean_perfs_runs)
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

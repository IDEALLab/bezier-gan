
"""
Optimize the airfoil shape in the latent space using Bayesian optimization, 
constrained on the running time

Author(s): Wei Chen (wchen459@umd.edu)
"""

from __future__ import division
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from bayesian_opt import optimize as optimize_latent
from functions import AirfoilGAN
from genetic_alg import generate_first_population, select, create_children, mutate_population
from utils import mean_err, create_dir


def optimize_overall(latent, noise0, perturb_type, perturb, n_eval, func):
    
    # Optimize in the latent+noise combined space
    n_best = 30
    n_random = 10
    n_children = 5
    chance_of_mutation = 0.1
    population_size = int((n_best+n_random)/2*n_children)
    x0 = np.append(latent, noise0)
    init_perf = func(x0)
    population = generate_first_population(x0, population_size, perturb_type, perturb)
    best_inds = [x0]
    best_perfs = [init_perf]
    opt_perfs = []
    i = 0
    print('Initial: x {} CL/CD {}'.format(x0, init_perf))
    while 1:
        breeders, best_perf, best_individual = select(population, n_best, n_random, func)
        best_inds.append(best_individual)
        best_perfs.append(best_perf)
        opt_perfs += [np.max(best_perfs)] * population_size # Best performance so far
        print('%d: fittest %.2f' % (i+1, best_perf))
        # No need to create next generation for the last generation
        if i < n_eval/population_size-1:
            next_generation = create_children(breeders, n_children)
            population = mutate_population(next_generation, chance_of_mutation, perturb_type, perturb)
            i += 1
        else:
            break
    
    opt_x = best_inds[np.argmax(best_perfs)]
    opt_airfoil = func.synthesize(opt_x)
    print('Optimal CL/CD: {}'.format(opt_perfs[-1]))
    
    return opt_x, opt_airfoil, opt_perfs


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Optimize')
    parser.add_argument('latent', type=int, default=3, help='latent dimension')
    parser.add_argument('noise', type=int, default=10, help='noise dimension')
    parser.add_argument('--n_runs', type=int, default=10, help='number of runs')
    parser.add_argument('--n_eval', type=int, default=1000, help='number of total evaluations per run')
    args = parser.parse_args()
    
    latent_dim = args.latent
    noise_dim = args.noise
    n_runs = args.n_runs
    n_eval = args.n_eval
    n_init_eval_latent = 10
    n_eval_latent = np.around(n_eval*latent_dim//(latent_dim+noise_dim), decimals=-2) # round to hundreds
    n_eval_overall = n_eval - n_eval_latent
    
    opt_x_runs = []
    opt_airfoil_runs = []
    opt_perfs_runs = []
    time_runs = []
    
    for i in range(n_runs):
                
        print('')
        print('######################################################')
        print('# Method: GAN-TSO {} {}'.format(args.latent, args.noise))
        print('# Run: {}/{}'.format(i+1, n_runs))
        print('######################################################')
        
#        # Randomly select from the trained GAN
#        main_directory = './beziergan/trained_gan/{}_{}'.format(latent_dim, noise_dim)
#        sub_directory = np.random.choice(next(os.walk(main_directory))[1])
#        model_directory = main_directory + '/' + sub_directory
#        print('Model ID: ' + sub_directory)
        model_directory = './beziergan/trained_gan/{}_{}/{}'.format(latent_dim, noise_dim, i)
        
        successful = False
        while not successful:
            try:
                start_time = time.time()
                # Optimize in the latent space
                func = AirfoilGAN(latent_dim, noise_dim, model_directory=model_directory)
                opt_latent, opt_airfoil, opt_perfs_latent = optimize_latent(n_eval_latent, n_init_eval_latent, func)
                # Optimize in the latent+noise combined space
                func = AirfoilGAN(latent_dim, noise_dim, model_directory=model_directory, latent=opt_latent)
                noise0 = np.zeros(noise_dim)
                perturb_type = 'absolute'
                perturb = np.append(0.1*np.ones(latent_dim), 1.0*np.ones(noise_dim))
                opt_x, opt_airfoil, opt_perfs_overall = optimize_overall(opt_latent, noise0, perturb_type, perturb, n_eval_overall, func)
                end_time = time.time()
                opt_x_runs.append(opt_x)
                opt_airfoil_runs.append(opt_airfoil)
                opt_perfs = opt_perfs_latent.tolist() + opt_perfs_overall
                opt_perfs_runs.append(opt_perfs)
                time_runs.append(end_time-start_time)
                successful = True
            except Exception as e:
                print(e)
    
    opt_x_runs = np.array(opt_x_runs)
    opt_airfoil_runs = np.array(opt_airfoil_runs)
    opt_perfs_runs = np.array(opt_perfs_runs)
    save_dir = 'results_opt/gan_bo_refine_{}_{}'.format(latent_dim, noise_dim)
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

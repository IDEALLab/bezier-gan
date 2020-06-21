"""
Compare preformance of methods within certain running time.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import argparse
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Optimize')
    parser.add_argument('--n_runs', type=int, default=10, help='number of runs')
    parser.add_argument('--n_eval', type=int, default=500, help='number of evaluations per run')
    args = parser.parse_args()
    
    n_runs = args.n_runs
    n_eval = args.n_eval
    
    ''' Call optimization '''
    for latent_dim in [2, 4, 6, 8, 10]:
        if not os.path.exists('./results_opt/gan_bo_refine_{}_10/opt_history.npy'.format(latent_dim)) or not os.path.exists('./results_opt/gan_bo_refine_{}_10/opt_airfoil.npy'.format(latent_dim)):
            os.system('python optimize_gan_bo_refine.py {} 10 --n_runs={} --n_eval={}'.format(latent_dim, args.n_runs, args.n_eval))
    if not os.path.exists('./results_opt/gan_bo_8_0/opt_history.npy') or not os.path.exists('./results_opt/gan_bo_8_0/opt_airfoil.npy'):
        os.system('python optimize_gan_bo.py 8 0 --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    if not os.path.exists('./results_opt/gan_bo_refine_8_20/opt_history.npy') or not os.path.exists('./results_opt/gan_bo_refine_8_20/opt_airfoil.npy'):
        os.system('python optimize_gan_bo_refine.py 8 20 --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    for latent_dim in [2, 4, 6, 8, 10]:
        if not os.path.exists('./results_opt/gan_bo_{}_10/opt_history.npy'.format(latent_dim)) or not os.path.exists('./results_opt/gan_bo_{}_10/opt_airfoil.npy'.format(latent_dim)):
            os.system('python optimize_gan_bo.py {} 10 --n_runs={} --n_eval={}'.format(latent_dim, args.n_runs, args.n_eval))
    for dim in [6, 8, 10, 12]:
        if not os.path.exists('./results_opt/generic_bo_{}/opt_history.npy'.format(dim)) or not os.path.exists('./results_opt/generic_bo_{}/opt_airfoil.npy'.format(dim)):
            os.system('python optimize_generic_bo.py {} --n_runs={} --n_eval={}'.format(dim, args.n_runs, args.n_eval))
    if not os.path.exists('./results_opt/generic_ga_8/opt_history.npy') or not os.path.exists('./results_opt/generic_ga_8/opt_airfoil.npy'):
        os.system('python optimize_generic_ga.py 8 --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    for dim in [8, 9, 10]:
        if not os.path.exists('./results_opt/svd_bo_{}/opt_history.npy'.format(dim)) or not os.path.exists('./results_opt/svd_bo_{}/opt_airfoil.npy'.format(dim)):
            os.system('python optimize_svd_bo.py {} --n_runs={} --n_eval={}'.format(dim, args.n_runs, args.n_eval))
    if not os.path.exists('./results_opt/svd_ga_9/opt_history.npy') or not os.path.exists('./results_opt/svd_ga_9/opt_airfoil.npy'):
        os.system('python optimize_svd_ga.py 9 --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    if not os.path.exists('./results_opt/nurbs_bo/opt_history.npy') or not os.path.exists('./results_opt/nurbs_bo/opt_airfoil.npy'):
        os.system('python optimize_nurbs_bo.py --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    if not os.path.exists('./results_opt/nurbs_ga/opt_history.npy') or not os.path.exists('./results_opt/nurbs_ga/opt_airfoil.npy'):
        os.system('python optimize_nurbs_ga.py --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    if not os.path.exists('./results_opt/ffd_bo/opt_history.npy') or not os.path.exists('./results_opt/ffd_bo/opt_airfoil.npy'):
        os.system('python optimize_ffd_bo.py --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    if not os.path.exists('./results_opt/ffd_ga/opt_history.npy') or not os.path.exists('./results_opt/ffd_ga/opt_airfoil.npy'):
        os.system('python optimize_ffd_ga.py --n_runs={} --n_eval={}'.format(args.n_runs, args.n_eval))
    
    ''' Plot history '''
    gan_bo_2_10_hist = np.load('results_opt/gan_bo_2_10/opt_history.npy')
    gan_bo_4_10_hist = np.load('results_opt/gan_bo_4_10/opt_history.npy')
    gan_bo_6_10_hist = np.load('results_opt/gan_bo_6_10/opt_history.npy')
    gan_bo_8_10_hist = np.load('results_opt/gan_bo_8_10/opt_history.npy')
    gan_bo_10_10_hist = np.load('results_opt/gan_bo_10_10/opt_history.npy')
    gan_bo_refine_2_10_hist = np.load('results_opt/gan_bo_refine_2_10/opt_history.npy')
    gan_bo_refine_4_10_hist = np.load('results_opt/gan_bo_refine_4_10/opt_history.npy')
    gan_bo_refine_6_10_hist = np.load('results_opt/gan_bo_refine_6_10/opt_history.npy')
    gan_bo_refine_8_10_hist = np.load('results_opt/gan_bo_refine_8_10/opt_history.npy')
    gan_bo_refine_10_10_hist = np.load('results_opt/gan_bo_refine_10_10/opt_history.npy')
    gan_bo_8_0_hist = np.load('results_opt/gan_bo_8_0/opt_history.npy')
    gan_bo_refine_8_20_hist = np.load('results_opt/gan_bo_refine_8_20/opt_history.npy')
    generic_bo_6_hist = np.load('results_opt/generic_bo_6/opt_history.npy')
    generic_bo_8_hist = np.load('results_opt/generic_bo_8/opt_history.npy')
    generic_bo_10_hist = np.load('results_opt/generic_bo_10/opt_history.npy')
    generic_bo_12_hist = np.load('results_opt/generic_bo_12/opt_history.npy')
    generic_ga_8_hist = np.load('results_opt/generic_ga_8/opt_history.npy')
    svd_bo_8_hist = np.load('results_opt/svd_bo_8/opt_history.npy')
    svd_bo_9_hist = np.load('results_opt/svd_bo_9/opt_history.npy')
    svd_bo_10_hist = np.load('results_opt/svd_bo_10/opt_history.npy')
    svd_ga_9_hist = np.load('results_opt/svd_ga_9/opt_history.npy')
    nurbs_bo_hist = np.load('results_opt/nurbs_bo/opt_history.npy')
    nurbs_ga_hist = np.load('results_opt/nurbs_ga/opt_history.npy')
    ffd_bo_hist = np.load('results_opt/ffd_bo/opt_history.npy')
    ffd_ga_hist = np.load('results_opt/ffd_ga/opt_history.npy')
    
    mean_gan_bo_2_10_hist = np.mean(gan_bo_2_10_hist, axis=0)
    std_gan_bo_2_10_hist = np.std(gan_bo_2_10_hist, axis=0)
    mean_gan_bo_4_10_hist = np.mean(gan_bo_4_10_hist, axis=0)
    std_gan_bo_4_10_hist = np.std(gan_bo_4_10_hist, axis=0)
    mean_gan_bo_6_10_hist = np.mean(gan_bo_6_10_hist, axis=0)
    std_gan_bo_6_10_hist = np.std(gan_bo_6_10_hist, axis=0)
    mean_gan_bo_8_10_hist = np.mean(gan_bo_8_10_hist, axis=0)
    std_gan_bo_8_10_hist = np.std(gan_bo_8_10_hist, axis=0)
    mean_gan_bo_10_10_hist = np.mean(gan_bo_10_10_hist, axis=0)
    std_gan_bo_10_10_hist = np.std(gan_bo_10_10_hist, axis=0)
    mean_gan_bo_refine_2_10_hist = np.mean(gan_bo_refine_2_10_hist, axis=0)
    std_gan_bo_refine_2_10_hist = np.std(gan_bo_refine_2_10_hist, axis=0)
    mean_gan_bo_refine_4_10_hist = np.mean(gan_bo_refine_4_10_hist, axis=0)
    std_gan_bo_refine_4_10_hist = np.std(gan_bo_refine_4_10_hist, axis=0)
    mean_gan_bo_refine_6_10_hist = np.mean(gan_bo_refine_6_10_hist, axis=0)
    std_gan_bo_refine_6_10_hist = np.std(gan_bo_refine_6_10_hist, axis=0)
    mean_gan_bo_refine_8_10_hist = np.mean(gan_bo_refine_8_10_hist, axis=0)
    std_gan_bo_refine_8_10_hist = np.std(gan_bo_refine_8_10_hist, axis=0)
    mean_gan_bo_refine_10_10_hist = np.mean(gan_bo_refine_10_10_hist, axis=0)
    std_gan_bo_refine_10_10_hist = np.std(gan_bo_refine_10_10_hist, axis=0)
    mean_gan_bo_8_0_hist = np.mean(gan_bo_8_0_hist, axis=0)
    std_gan_bo_8_0_hist = np.std(gan_bo_8_0_hist, axis=0)
    mean_gan_bo_refine_8_20_hist = np.mean(gan_bo_refine_8_20_hist, axis=0)
    std_gan_bo_refine_8_20_hist = np.std(gan_bo_refine_8_20_hist, axis=0)
    mean_generic_bo_6_hist = np.mean(generic_bo_6_hist, axis=0)
    std_generic_bo_6_hist = np.std(generic_bo_6_hist, axis=0)
    mean_generic_bo_8_hist = np.mean(generic_bo_8_hist, axis=0)
    std_generic_bo_8_hist = np.std(generic_bo_8_hist, axis=0)
    mean_generic_bo_10_hist = np.mean(generic_bo_10_hist, axis=0)
    std_generic_bo_10_hist = np.std(generic_bo_10_hist, axis=0)
    mean_generic_bo_12_hist = np.mean(generic_bo_12_hist, axis=0)
    std_generic_bo_12_hist = np.std(generic_bo_12_hist, axis=0)
    mean_generic_ga_8_hist = np.mean(generic_ga_8_hist, axis=0)
    std_generic_ga_8_hist = np.std(generic_ga_8_hist, axis=0)
    mean_svd_bo_8_hist = np.mean(svd_bo_8_hist, axis=0)
    std_svd_bo_8_hist = np.std(svd_bo_8_hist, axis=0)
    mean_svd_bo_9_hist = np.mean(svd_bo_9_hist, axis=0)
    std_svd_bo_9_hist = np.std(svd_bo_9_hist, axis=0)
    mean_svd_bo_10_hist = np.mean(svd_bo_10_hist, axis=0)
    std_svd_bo_10_hist = np.std(svd_bo_10_hist, axis=0)
    mean_svd_ga_9_hist = np.mean(svd_ga_9_hist, axis=0)
    std_svd_ga_9_hist = np.std(svd_ga_9_hist, axis=0)
    mean_nurbs_bo_hist = np.mean(nurbs_bo_hist, axis=0)
    std_nurbs_bo_hist = np.std(nurbs_bo_hist, axis=0)
    mean_nurbs_ga_hist = np.mean(nurbs_ga_hist, axis=0)
    std_nurbs_ga_hist = np.std(nurbs_ga_hist, axis=0)
    mean_ffd_bo_hist = np.mean(ffd_bo_hist, axis=0)
    std_ffd_bo_hist = np.std(ffd_bo_hist, axis=0)
    mean_ffd_ga_hist = np.mean(ffd_ga_hist, axis=0)
    std_ffd_ga_hist = np.std(ffd_ga_hist, axis=0)
    
    linestyles = ['-', '--', ':', '-.', (0, (5,1,1,1,1,1)), (0, (1,4))]
    lss = itertools.cycle(linestyles)
    iters = np.arange(n_eval+1, dtype=int)
    
    diff_2 = gan_bo_refine_2_10_hist - gan_bo_2_10_hist
    mean_diff_2 = np.mean(diff_2, axis=0)
    std_diff_2 = np.std(diff_2, axis=0)
    diff_4 = gan_bo_refine_4_10_hist - gan_bo_4_10_hist
    mean_diff_4 = np.mean(diff_4, axis=0)
    std_diff_4 = np.std(diff_4, axis=0)
    diff_6 = gan_bo_refine_6_10_hist - gan_bo_6_10_hist
    mean_diff_6 = np.mean(diff_6, axis=0)
    std_diff_6 = np.std(diff_6, axis=0)
    diff_8 = gan_bo_refine_8_10_hist - gan_bo_8_10_hist
    mean_diff_8 = np.mean(diff_8, axis=0)
    std_diff_8 = np.std(diff_8, axis=0)
    diff_10 = gan_bo_refine_10_10_hist - gan_bo_10_10_hist
    mean_diff_10 = np.mean(diff_10, axis=0)
    std_diff_10 = np.std(diff_10, axis=0)
    
    plt.figure(figsize=(10,5))
    plt.plot(iters, mean_gan_bo_6_10_hist, ls=next(lss), label='OSO', c='k', alpha=.7)
    plt.fill_between(iters, mean_gan_bo_6_10_hist-std_gan_bo_6_10_hist, mean_gan_bo_6_10_hist+std_gan_bo_6_10_hist, alpha=.2, color='0.6')
    plt.plot(iters, mean_gan_bo_refine_6_10_hist, ls=next(lss), label='TSO', c='k', alpha=.7)
    plt.fill_between(iters, mean_gan_bo_refine_6_10_hist-std_gan_bo_refine_6_10_hist, mean_gan_bo_refine_6_10_hist+std_gan_bo_refine_6_10_hist, alpha=.2, color='0.6')
    plt.legend(frameon=False, title='Latent dim.')
    plt.title('Performance History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel(r'$C_L/C_D$')
#    plt.xticks(np.linspace(0, max_n_eval+1, 5, dtype=int))
    plt.tight_layout()
    plt.savefig('results_opt/opt_history_refine_6.svg')
    plt.savefig('results_opt/opt_history_refine_6.pdf', dpi=600)
    plt.close()
    
    list_latent_dim = [2, 4, 6, 8, 10]
    list_mean = [mean_gan_bo_2_10_hist[-1], 
                 mean_gan_bo_4_10_hist[-1], 
                 mean_gan_bo_6_10_hist[-1], 
                 mean_gan_bo_8_10_hist[-1], 
                 mean_gan_bo_10_10_hist[-1]]
    list_std = [std_gan_bo_2_10_hist[-1], 
                std_gan_bo_4_10_hist[-1], 
                std_gan_bo_6_10_hist[-1], 
                std_gan_bo_8_10_hist[-1], 
                std_gan_bo_10_10_hist[-1]]
    list_mean_refine = [mean_gan_bo_refine_2_10_hist[-1], 
                        mean_gan_bo_refine_4_10_hist[-1], 
                        mean_gan_bo_refine_6_10_hist[-1], 
                        mean_gan_bo_refine_8_10_hist[-1], 
                        mean_gan_bo_refine_10_10_hist[-1]]
    list_std_refine = [std_gan_bo_refine_2_10_hist[-1], 
                       std_gan_bo_refine_4_10_hist[-1], 
                       std_gan_bo_refine_6_10_hist[-1], 
                       std_gan_bo_refine_8_10_hist[-1], 
                       std_gan_bo_refine_10_10_hist[-1]]
    plt.figure(figsize=(5,5))
    plt.bar(np.array(list_latent_dim)-.3, list_mean, width=.6, yerr=list_std, label='One-stage opt.', color='0.7')
    plt.bar(np.array(list_latent_dim)+.3, list_mean_refine, width=.6, yerr=list_std_refine, label='Two-stage opt.', color='0.3')
    plt.legend(frameon=False)
    plt.ylim([50, 350])
    plt.xticks(list_latent_dim)
    plt.xlabel('Latent dimension')
    plt.ylabel(r'Max. $C_L/C_D$')
    plt.tight_layout()
    plt.savefig('results_opt/opt_refine.svg')
    plt.savefig('results_opt/opt_refine.pdf', dpi=600)
    plt.close()
    
    lss = itertools.cycle(linestyles)
    
    plt.figure(figsize=(7,5))
    plt.plot(iters, mean_diff_2, ls=next(lss), label='2', c='k', alpha=.7)
    plt.fill_between(iters, mean_diff_2-std_diff_2, mean_diff_2+std_diff_2, alpha=.2, color='0.6')
#    plt.plot(iters, mean_diff_4, ls=next(lss), label='4', c='k', alpha=.7)
#    plt.fill_between(iters, mean_diff_4-std_diff_4, mean_diff_4+std_diff_4, alpha=.2, color='0.6')
#    plt.plot(iters, mean_diff_6, ls=next(lss), label='6', c='k', alpha=.7)
#    plt.fill_between(iters, mean_diff_6-std_diff_6, mean_diff_6+std_diff_6, alpha=.2, color='0.6')
    plt.plot(iters, mean_diff_8, ls=next(lss), label='8', c='k', alpha=.7)
    plt.fill_between(iters, mean_diff_8-std_diff_8, mean_diff_8+std_diff_8, alpha=.2, color='0.6')
#    plt.plot(iters, mean_diff_10, ls=next(lss), label='10', c='k', alpha=.7)
#    plt.fill_between(iters, mean_diff_10-std_diff_10, mean_diff_10+std_diff_10, alpha=.2, color='0.6')
    plt.axhline(y=0., color='r', ls=next(lss), alpha=.7)
    plt.legend(frameon=False, title='Latent dim.')
    plt.title('Performance Improvement History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel(r'$\Delta$ ($C_L/C_D$)')
#    plt.xticks(np.linspace(0, max_n_eval+1, 5, dtype=int))
    plt.tight_layout()
    plt.savefig('results_opt/opt_history_refine.svg')
    plt.savefig('results_opt/opt_history_refine.pdf', dpi=600)
    plt.close()
    
    lss = itertools.cycle(linestyles)
    
    plt.figure(figsize=(10,5))
    plt.plot(iters, mean_gan_bo_refine_8_10_hist, ls=next(lss), label=r'B$\acute{e}$zier-GAN+TSO', c='k', alpha=.7)
    plt.fill_between(iters, mean_gan_bo_refine_8_10_hist-std_gan_bo_refine_8_10_hist, mean_gan_bo_refine_8_10_hist+std_gan_bo_refine_8_10_hist, alpha=.2, color='0.6')
    plt.plot(iters, mean_generic_bo_8_hist, ls=next(lss), label='GMDV+EGO', c='k', alpha=.7)
    plt.fill_between(iters, mean_generic_bo_8_hist-std_generic_bo_8_hist, mean_generic_bo_8_hist+std_generic_bo_8_hist, alpha=.2, color='0.6')
#    plt.plot(iters, mean_generic_ga_8_hist, ls=next(lss), label='GMDV+GA')
#    plt.fill_between(iters, mean_generic_ga_8_hist-std_generic_ga_8_hist, mean_generic_ga_8_hist+std_generic_ga_8_hist, alpha=.2, color='0.6')
    plt.plot(iters, mean_svd_bo_9_hist, ls=next(lss), label='SVD+EGO', c='k', alpha=.7)
    plt.fill_between(iters, mean_svd_bo_9_hist-std_svd_bo_9_hist, mean_svd_bo_9_hist+std_svd_bo_9_hist, alpha=.2, color='0.6')
#    plt.plot(iters, mean_svd_ga_9_hist, ls=next(lss), label='SVD+GA', c='k', alpha=.7)
#    plt.fill_between(iters, mean_svd_ga_9_hist-std_svd_ga_9_hist, mean_svd_ga_9_hist+std_svd_ga_9_hist, alpha=.2, color='0.6')
    plt.plot(iters, mean_ffd_bo_hist, ls=next(lss), label='FFD+EGO', c='k', alpha=.7)
    plt.fill_between(iters, mean_ffd_bo_hist-std_ffd_bo_hist, mean_ffd_bo_hist+std_ffd_bo_hist, alpha=.2, color='0.6')
    plt.plot(iters, mean_ffd_ga_hist, ls=next(lss), label='FFD+GA', c='k', alpha=.7)
    plt.fill_between(iters, mean_ffd_ga_hist-std_ffd_ga_hist, mean_ffd_ga_hist+std_ffd_ga_hist, alpha=.2, color='0.6')
    plt.legend(frameon=False)
    plt.title('Optimization History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel(r'$C_L/C_D$')
    plt.ylim(ymax=300)
#    plt.xticks(np.linspace(0, max_n_eval+1, 5, dtype=int))
    plt.tight_layout()
    plt.savefig('results_opt/opt_history_conventional.svg')
    plt.savefig('results_opt/opt_history_conventional.pdf', dpi=600)
    plt.close()
    
    lss = itertools.cycle(linestyles)
    
    plt.figure(figsize=(10,5))
    plt.plot(iters, mean_gan_bo_refine_2_10_hist, ls=next(lss), label='2', c='k', alpha=.7)
    plt.fill_between(iters, mean_gan_bo_refine_2_10_hist-std_gan_bo_refine_2_10_hist, mean_gan_bo_refine_2_10_hist+std_gan_bo_refine_2_10_hist, alpha=.2, color='0.6')
    plt.plot(iters, mean_gan_bo_refine_4_10_hist, ls=next(lss), label='4', c='k', alpha=.7)
    plt.fill_between(iters, mean_gan_bo_refine_4_10_hist-std_gan_bo_refine_4_10_hist, mean_gan_bo_refine_4_10_hist+std_gan_bo_refine_4_10_hist, alpha=.2, color='0.6')
    plt.plot(iters, mean_gan_bo_refine_6_10_hist, ls=next(lss), label='6', c='k', alpha=.7)
    plt.fill_between(iters, mean_gan_bo_refine_6_10_hist-std_gan_bo_refine_6_10_hist, mean_gan_bo_refine_6_10_hist+std_gan_bo_refine_6_10_hist, alpha=.2, color='0.6')
    plt.plot(iters, mean_gan_bo_refine_8_10_hist, ls=next(lss), label='8', c='k', alpha=.7)
    plt.fill_between(iters, mean_gan_bo_refine_8_10_hist-std_gan_bo_refine_8_10_hist, mean_gan_bo_refine_8_10_hist+std_gan_bo_refine_8_10_hist, alpha=.2, color='0.6')
    plt.plot(iters, mean_gan_bo_refine_10_10_hist, ls=next(lss), label='10', c='k', alpha=.7)
    plt.fill_between(iters, mean_gan_bo_refine_10_10_hist-std_gan_bo_refine_10_10_hist, mean_gan_bo_refine_10_10_hist+std_gan_bo_refine_10_10_hist, alpha=.2, color='0.6')
    plt.legend(frameon=False, title='Latent dim.')
    plt.title('Optimization History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel(r'$C_L/C_D$')
    plt.ylim(ymax=300)
#    plt.xticks(np.linspace(0, max_n_eval+1, 5, dtype=int))
    plt.tight_layout()
    plt.savefig('results_opt/opt_history_latent.svg')
    plt.savefig('results_opt/opt_history_latent.pdf', dpi=600)
    plt.close()
    
    lss = itertools.cycle(linestyles)
    
    plt.figure(figsize=(10,5))
    plt.plot(iters, mean_gan_bo_8_0_hist, ls=next(lss), label='0', c='k', alpha=.7)
    plt.fill_between(iters, mean_gan_bo_8_0_hist-std_gan_bo_8_0_hist, mean_gan_bo_8_0_hist+std_gan_bo_8_0_hist, alpha=.2, color='0.6')
    plt.plot(iters, mean_gan_bo_refine_8_10_hist, ls=next(lss), label='10', c='k', alpha=.7)
    plt.fill_between(iters, mean_gan_bo_refine_8_10_hist-std_gan_bo_refine_8_10_hist, mean_gan_bo_refine_8_10_hist+std_gan_bo_refine_8_10_hist, alpha=.2, color='0.6')
    plt.plot(iters, mean_gan_bo_refine_8_20_hist, ls=next(lss), label='20', c='k', alpha=.7)
    plt.fill_between(iters, mean_gan_bo_refine_8_20_hist-std_gan_bo_refine_8_20_hist, mean_gan_bo_refine_8_20_hist+std_gan_bo_refine_8_20_hist, alpha=.2, color='0.6')
    plt.legend(frameon=False, title='Noise dim.')
    plt.title('Optimization History')
    plt.xlabel('Number of Evaluations')
    plt.ylabel(r'$C_L/C_D$')
    plt.ylim(ymax=300)
#    plt.xticks(np.linspace(0, max_n_eval+1, 5, dtype=int))
    plt.tight_layout()
    plt.savefig('results_opt/opt_history_noise.svg')
    plt.savefig('results_opt/opt_history_noise.pdf', dpi=600)
    plt.close()
    
    ''' Plot optimal solutions '''
    gan_bo_4_10_opt = np.load('results_opt/gan_bo_4_10/opt_airfoil.npy')
    gan_bo_refine_2_10_opt = np.load('results_opt/gan_bo_refine_2_10/opt_airfoil.npy')
    gan_bo_refine_4_10_opt = np.load('results_opt/gan_bo_refine_4_10/opt_airfoil.npy')
    gan_bo_refine_6_10_opt = np.load('results_opt/gan_bo_refine_6_10/opt_airfoil.npy')
    gan_bo_refine_8_10_opt = np.load('results_opt/gan_bo_refine_8_10/opt_airfoil.npy')
    gan_bo_refine_10_10_opt = np.load('results_opt/gan_bo_refine_10_10/opt_airfoil.npy')
    gan_bo_8_0_opt = np.load('results_opt/gan_bo_8_0/opt_airfoil.npy')
    gan_bo_refine_8_10_opt = np.load('results_opt/gan_bo_refine_8_10/opt_airfoil.npy')
    gan_bo_refine_8_20_opt = np.load('results_opt/gan_bo_refine_8_20/opt_airfoil.npy')
    generic_bo_8_opt = np.load('results_opt/generic_bo_8/opt_airfoil.npy')
    generic_ga_8_opt = np.load('results_opt/generic_ga_8/opt_airfoil.npy')
    svd_bo_9_opt = np.load('results_opt/svd_bo_9/opt_airfoil.npy')
    svd_ga_9_opt = np.load('results_opt/svd_ga_9/opt_airfoil.npy')
    nurbs_bo_opt = np.load('results_opt/nurbs_bo/opt_airfoil.npy')
    nurbs_ga_opt = np.load('results_opt/nurbs_ga/opt_airfoil.npy')
    ffd_bo_opt = np.load('results_opt/ffd_bo/opt_airfoil.npy')
    ffd_ga_opt = np.load('results_opt/ffd_ga/opt_airfoil.npy')
    
    # Separate plots
    def subplot_airfoil(position, airfoils, title=None):
        if type(position) == tuple:
            ax = plt.subplot(*position)
        else:
            ax = plt.subplot(position)
        n = airfoils.shape[0]
        for airfoil in airfoils:
            ax.plot(airfoil[:,0], airfoil[:,1], '-', c='k', alpha=1.0/n)
        if title is not None:
            ax.title.set_text(title)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.2, 0.2])
        ax.axis('equal')
        ax.axis('off')
#        plt.set_xlabel('x')
#        plt.set_ylabel('y')
    
    def plot_opt_airfoils(opt_airfoils, opt_hist, figname):
        plt.figure(figsize=(25, 4))
        subplot_airfoil(251, opt_airfoils[0:1], 'CL/CD={:.2f}'.format(opt_hist[0,-1]))
        subplot_airfoil(252, opt_airfoils[1:2], 'CL/CD={:.2f}'.format(opt_hist[1,-1]))
        subplot_airfoil(253, opt_airfoils[2:3], 'CL/CD={:.2f}'.format(opt_hist[2,-1]))
        subplot_airfoil(254, opt_airfoils[3:4], 'CL/CD={:.2f}'.format(opt_hist[3,-1]))
        subplot_airfoil(255, opt_airfoils[4:5], 'CL/CD={:.2f}'.format(opt_hist[4,-1]))
        subplot_airfoil(256, opt_airfoils[5:6], 'CL/CD={:.2f}'.format(opt_hist[5,-1]))
        subplot_airfoil(257, opt_airfoils[6:7], 'CL/CD={:.2f}'.format(opt_hist[6,-1]))
        subplot_airfoil(258, opt_airfoils[7:8], 'CL/CD={:.2f}'.format(opt_hist[7,-1]))
        subplot_airfoil(259, opt_airfoils[8:9], 'CL/CD={:.2f}'.format(opt_hist[8,-1]))
        subplot_airfoil((2,5,10), opt_airfoils[9:10], 'CL/CD={:.2f}'.format(opt_hist[9,-1]))
        plt.tight_layout()
        plt.savefig(figname)
        plt.close()
        
    plot_opt_airfoils(gan_bo_refine_8_10_opt, gan_bo_refine_8_10_hist, 'results_opt/opt_airfoils_gan_bo_refine_8_10.svg')
    plot_opt_airfoils(gan_bo_4_10_opt, gan_bo_4_10_hist, 'results_opt/opt_airfoils_gan_bo_4_10.svg')
    plot_opt_airfoils(generic_bo_8_opt, generic_bo_8_hist, 'results_opt/opt_airfoils_generic_bo_8.svg')
    plot_opt_airfoils(generic_ga_8_opt, generic_ga_8_hist, 'results_opt/opt_airfoils_generic_ga_8.svg')
    plot_opt_airfoils(svd_bo_9_opt, svd_bo_9_hist, 'results_opt/opt_airfoils_svd_bo_9.svg')
    plot_opt_airfoils(svd_ga_9_opt, svd_ga_9_hist, 'results_opt/opt_airfoils_svd_ga_9.svg')
    plot_opt_airfoils(nurbs_bo_opt, nurbs_bo_hist, 'results_opt/opt_airfoils_nurbs_bo.svg')
    plot_opt_airfoils(nurbs_ga_opt, nurbs_ga_hist, 'results_opt/opt_airfoils_nurbs_ga.svg')
    plot_opt_airfoils(ffd_bo_opt, ffd_bo_hist, 'results_opt/opt_airfoils_ffd_bo.svg')
    plot_opt_airfoils(ffd_ga_opt, ffd_ga_hist, 'results_opt/opt_airfoils_ffd_ga.svg')
    
    plt.figure(figsize=(15, 4))
    subplot_airfoil(241, gan_bo_refine_8_10_opt, r'B$\acute{e}$zier-GAN+TSO')
    subplot_airfoil(242, ffd_bo_opt, 'FFD+EGO')
    subplot_airfoil(243, ffd_ga_opt, 'FFD+GA')
    subplot_airfoil(244, generic_bo_8_opt, 'GMDV+EGO')
    subplot_airfoil(245, generic_ga_8_opt, 'GMDV+GA')
    subplot_airfoil(246, svd_bo_9_opt, 'SVD+EGO')
    subplot_airfoil(247, svd_bo_9_opt, 'SVD+GA')
    plt.tight_layout()
    plt.savefig('results_opt/opt_airfoils_conventional.svg')
    plt.close()
    
    plt.figure(figsize=(10, 2))
    subplot_airfoil(121, gan_bo_refine_8_10_opt, 'Two-stage optimization')
    subplot_airfoil(122, gan_bo_4_10_opt, 'One-stage optimization')
    plt.tight_layout()
    plt.savefig('results_opt/opt_airfoils_refine.svg')
    plt.close()
    
    plt.figure(figsize=(20, 2))
    subplot_airfoil(151, gan_bo_refine_2_10_opt, '2')
    subplot_airfoil(152, gan_bo_refine_4_10_opt, '4')
    subplot_airfoil(153, gan_bo_refine_6_10_opt, '6')
    subplot_airfoil(154, gan_bo_refine_8_10_opt, '8')
    subplot_airfoil(155, gan_bo_refine_10_10_opt, '10')
    plt.tight_layout()
    plt.savefig('results_opt/opt_airfoils_latent.svg')
    plt.close()
    
    plt.figure(figsize=(15, 2))
    subplot_airfoil(131, gan_bo_8_0_opt, '0')
    subplot_airfoil(132, gan_bo_refine_8_10_opt, '10')
    subplot_airfoil(133, gan_bo_refine_8_20_opt, '20')
    plt.tight_layout()
    plt.savefig('results_opt/opt_airfoils_noise.svg')
    plt.close()

"""
Trains a BezierGAN, and visulizes results

Author(s): Wei Chen (wchen459@umd.edu)
"""

import argparse
import numpy as np

from gan import GAN
from mmd import ci_mmd
from consistency import ci_cons

import sys 
sys.path.append('..')
from shape_plot import plot_samples, plot_grid
from utils import ElapsedTimer, train_test_plit


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default='train', help='train or evaluate')
    parser.add_argument('latent', type=int, default=3, help='latent dimension')
    parser.add_argument('noise', type=int, default=10, help='noise dimension')
    parser.add_argument('--model_id', type=int, default=None, help='model ID')
    parser.add_argument('--save_interval', type=int, default=500, help='save interval')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
    latent_dim = args.latent
    noise_dim = args.noise
    bezier_degree = 31
    train_steps = 10000
    batch_size = 32
    symm_axis = None
    bounds = (0., 1.)
    
    # Read dataset
    data_fname = '../data/airfoil_interp.npy'
    X = np.load(data_fname)
    
    print('Plotting training samples ...')
    samples = X[np.random.choice(range(X.shape[0]), size=36)]
#    plot_samples(None, samples, scatter=True, symm_axis=symm_axis, s=1.5, alpha=.7, c='k', fname='samples')
    plot_samples(None, samples, scale=1.0, scatter=False, symm_axis=symm_axis, lw=1.2, alpha=.7, c='k', fname='samples')
    
    # Split training and test data
    X_train, X_test = train_test_plit(X, split=0.8)
    
    # Train
    directory = './trained_gan/{}_{}'.format(latent_dim, noise_dim)
    if args.model_id is not None:
        directory += '/{}'.format(args.model_id)
    model = GAN(latent_dim, noise_dim, X_train.shape[1], bezier_degree, bounds)
    if args.mode == 'train':
        timer = ElapsedTimer()
        model.train(X_train, batch_size=batch_size, train_steps=train_steps, save_interval=args.save_interval, directory=directory)
        elapsed_time = timer.elapsed_time()
        runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
        print(runtime_mesg)
        runtime_file = open('{}/runtime.txt'.format(directory), 'w')
        runtime_file.write('%s\n' % runtime_mesg)
        runtime_file.close()
    else:
        model.restore(directory=directory)
    
    print('Plotting synthesized shapes ...')
    points_per_axis = 5
    plot_grid(points_per_axis, gen_func=model.synthesize, d=latent_dim, bounds=bounds, scale=1.0, scatter=False, symm_axis=symm_axis, 
              alpha=.7, lw=1.2, c='k', fname='{}/synthesized'.format(directory))
    def synthesize_noise(noise):
        return model.synthesize(0.5*np.ones((points_per_axis**2, latent_dim)), noise)
    plot_grid(points_per_axis, gen_func=synthesize_noise, d=noise_dim, bounds=(-1., 1.), scale=1.0, scatter=False, symm_axis=symm_axis, 
              alpha=.7, lw=1.2, c='k', fname='{}/synthesized_noise'.format(directory))
    
    n_runs = 10
    
    mmd_mean, mmd_err = ci_mmd(n_runs, model.synthesize, X_test)
    cons_mean, cons_err = ci_cons(n_runs, model.synthesize, latent_dim, bounds)
    
    results_mesg_1 = 'Maximum mean discrepancy: %.4f +/- %.4f' % (mmd_mean, mmd_err)
    results_mesg_2 = 'Consistency: %.3f +/- %.3f' % (cons_mean, cons_err)
    
    results_file = open('{}/results.txt'.format(directory), 'w')
    
    print(results_mesg_1)
    results_file.write('%s\n' % results_mesg_1)
    print(results_mesg_2)
    results_file.write('%s\n' % results_mesg_2)
        
    results_file.close()

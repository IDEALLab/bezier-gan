
"""
Plot evaluation metrics for GANs with different latent/noise dimensions.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import shutil
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from gan import GAN
from mmd import maximum_mean_discrepancy
from consistency import consistency

import sys 
sys.path.append('..')
from utils import mean_err, get_n_vars, train_test_plit


if __name__ == "__main__":
    
    latent_dims = [2, 4, 6, 8, 10]
    noise_dims = [0, 10, 20]
    bezier_degree = 31
    bounds = (0., 1.)
    n_runs = 10
    
    # Run experiments
    for noise_dim in noise_dims:
        for latent_dim in latent_dims:
            for i in range(n_runs):
                directory = 'trained_gan/{}_{}/{}'.format(latent_dim, noise_dim, i)
                    
#                while True:
#                    # Train if the model does not exist
#                    if not os.path.exists(directory+'/results.txt') or not os.path.exists(directory+'/model.meta'):
#                        print('Training for {}-{}/{} ...'.format(latent_dim, noise_dim, i))
#                        os.system('python train_gan.py train {} {} --model_id={} --save_interval=0'.format(latent_dim, noise_dim, i))
#                    # Test if model is valid
#                    print("Examing model from '{}' ...".format(directory))
#                    os.system('python eval_gan.py {} {} --model_id={}'.format(latent_dim, noise_dim, i))
#                    valid = np.load('../tmp/eval_manifold_validity.npy')
#                    print(valid)
#                    if valid:
#                        break
#                    else:
#                        # Delete if model is not valid
#                        print("Removing model from '{}' ...".format(directory))
#                        shutil.rmtree(directory)
#                
#                # Test whether model is valid
#                print("Examing model from '{}' ...".format(directory))
#                os.system('python eval_gan.py {} {} --model_id={}'.format(latent_dim, noise_dim, i))
#                valid = np.load('../tmp/eval_manifold_validity.npy')
#                print(valid)
#                if not valid:
#                    # Delete if model is not valid
#                    print("Removing model from '{}' ...".format(directory))
#                    shutil.rmtree(directory)
                
                # Train if the model does not exist
                if not os.path.exists(directory+'/results.txt') or not os.path.exists(directory+'/model.meta'):
                    print('Training for {}-{}/{} ...'.format(latent_dim, noise_dim, i))
                    os.system('python train_gan.py train {} {} --model_id={} --save_interval=0'.format(latent_dim, noise_dim, i))
                            
    # Read dataset
    data_fname = '../data/airfoil_interp.npy'
    X = np.load(data_fname)
    # Split training and test data
    X_train, X_test = train_test_plit(X, split=0.8)
    
    fig = plt.figure(figsize=(6,5))
    ax_mmd = fig.add_subplot(111)
    
    width = 0.5
    p_list = [-1., 0., 1.]
    c_list = ['0.8', '0.5', '0.2']
    
    for j, noise_dim in enumerate(noise_dims):
        
        list_mmd_mean = []
        list_mmd_err = []
        
        for latent_dim in latent_dims:
            
            list_mmd = []
            
            for i in range(n_runs):
            
                print('######################################################')
                print('{}-{}'.format(latent_dim, noise_dim))
                print('{}/{}'.format(i+1, n_runs))
                print('######################################################')
                
                directory = 'trained_gan/{}_{}/{}'.format(latent_dim, noise_dim, i)
                model = GAN(latent_dim, noise_dim, X_train.shape[1], bezier_degree, bounds)
                model.restore(directory=directory)
                
                mmd = maximum_mean_discrepancy(model.synthesize, X_test)
                list_mmd.append(mmd)
                
                print(get_n_vars())
                tf.keras.backend.clear_session()
                print(get_n_vars())
            
            mmd_mean, mmd_err = mean_err(list_mmd)
            
            list_mmd_mean.append(mmd_mean)
            list_mmd_err.append(mmd_err)
            
        ax_mmd.bar(np.array(latent_dims)+p_list[j]*width, list_mmd_mean, width, yerr=list_mmd_err, label=str(noise_dim), color=c_list[j])
            
    ax_mmd.legend(frameon=False, title='Noise dim.')
    ax_mmd.set_xticks(latent_dims)
    ax_mmd.set_xlabel('Latent dimension')
    ax_mmd.set_ylabel('MMD')
    
    plt.tight_layout()
    plt.savefig('trained_gan/mmd.svg')
    plt.savefig('trained_gan/mmd.pdf', dpi=600)
    plt.close()
"""
Evaluates the validity of an airfoil manifold

Author(s): Wei Chen (wchen459@umd.edu)
"""

import argparse
import numpy as np
from pyDOE import lhs
import tensorflow as tf

import sys 
sys.path.append('..')
from functions import AirfoilGAN
from utils import get_n_vars, create_dir


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Evaluate manifold validity')
    parser.add_argument('latent', type=int, default=3, help='latent dimension')
    parser.add_argument('noise', type=int, default=10, help='noise dimension')
    parser.add_argument('--model_id', type=int, default=None, help='model ID')
    args = parser.parse_args()
    
    latent_dim = args.latent
    noise_dim = args.noise
    i = args.model_id
    
    print(get_n_vars())
    tf.keras.backend.clear_session()
    print(get_n_vars())
    
    directory = './trained_gan/{}_{}/{}'.format(latent_dim, noise_dim, i)
    func = AirfoilGAN(latent_dim, noise_dim, model_directory=directory, config_fname='../op_conditions.ini')
    xs = lhs(latent_dim, samples=10*latent_dim, criterion='cm')
    ys = func(xs)
    vs = func.is_feasible(xs)
    valid = (np.sum(vs)/len(vs) >= 0.3)
    
    create_dir('../tmp')
    np.save('../tmp/eval_manifold_validity.npy', valid)
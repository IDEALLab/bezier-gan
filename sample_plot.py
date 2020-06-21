"""
Sample plot.

Author(s): Wei Chen (wchen459@umd.edu)
"""

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

import functions
from shape_plot import plot_shape
from utils import gen_grid


def plot(func, fig, points_per_axis, position, title):
    n_airfoils = points_per_axis**2
    airfoils = func.sample_airfoil(n_airfoils)
    ax = fig.add_subplot(2, 2, position)
    Z = gen_grid(2, points_per_axis, 0., 1.*points_per_axis)
    for (i, z) in enumerate(Z):
        plot_shape(airfoils[i], z[0], .5*z[1], ax, 1.1, False, None, c='k', lw=1.2)
    ax.set_title(title)
    plt.axis('off')
    plt.axis('equal')
    return fig


if __name__ == "__main__":
    
    points_per_axis = 4
    fig = plt.figure(figsize=(9, 6))
    
    ''' BezierGAN '''
    latent_dim = 8
    noise_dim = 10
    model_directory = './beziergan/trained_gan/{}_{}/0'.format(latent_dim, noise_dim)
    func = functions.AirfoilGAN(latent_dim, noise_dim, model_directory, full=True)
    fig = plot(func, fig, points_per_axis, 1, r'B$\acute{e}$zier-GAN')
    
    ''' SVD '''
    latent_dim = 9
    func = functions.AirfoilSVD(latent_dim)
    fig = plot(func, fig, points_per_axis, 2, 'SVD')
    
    ''' GMDV '''
    dim = 8
    func = functions.AirfoilGeneric(dim)
    fig = plot(func, fig, points_per_axis, 3, 'GMDV')
        
    ''' FFD '''
    dim = 12
    func = functions.AirfoilFFD(m=dim//3, n=3, initial_path='initial_airfoil/naca0012_uniform_192.dat')
    fig = plot(func, fig, points_per_axis, 4, 'FFD')
    
    plt.tight_layout()
    plt.savefig('samples.svg')
    plt.savefig('samples.pdf')
    plt.close()
        
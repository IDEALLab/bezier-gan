import numpy as np
from matplotlib import pyplot as plt
from functions import *


if __name__ == "__main__":
    
    n_rows = 5
    n_cols = 7
    n_airfoils = n_rows*n_cols
    
    dim = 10
    
#    parameterization = 'generic'
#    af = AirfoilGeneric(dim)
    
    parameterization = 'svd'
    af = AirfoilSVD(dim)
    
#    parameterization = 'ffd'
#    af = AirfoilFFD()
    
#    parameterization = 'nurbs'
#    af = AirfoilNurbs()
    
    plt.figure(figsize=(n_cols*4, n_rows*2))
    for i in range(n_airfoils):
        x = np.random.uniform(af.bounds[:,0], af.bounds[:,1])
        airfoil = af.synthesize(x)
        plt.subplot(n_rows, n_cols, i+1)
        plt.plot(airfoil[:,0], airfoil[:,1], '.-', ms=1, lw=1, alpha=0.5)
        plt.ylim([-0.2, 0.2])
        plt.axis('equal')
    plt.tight_layout()
#    plt.show()
    plt.savefig('{}/samples.svg'.format(parameterization))
    plt.close()
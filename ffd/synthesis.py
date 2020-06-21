from __future__ import division
#import numpy as np
import autograd.numpy as np
import scipy.special


def bernstein(xx, m, i):
    b = scipy.special.binom(m, i) * xx**i * (1-xx)**(m-i)
    return b

def synthesize(x, airfoil0, m, n, Px):
    '''
    Reference:
        Masters, D. A., Taylor, N. J., Rendall, T. C. S., Allen, C. B., & Poole, D. J. (2017). 
        Geometric comparison of aerofoil shape parameterization methods. AIAA Journal, 1575-1589.
    '''
    P = np.stack((Px, x.reshape(n,m)), axis=-1)
    xx = airfoil0[:,0]
    z_min = np.min(airfoil0[:,1])
    z_max = np.max(airfoil0[:,1])
    zz = (airfoil0[:,1]-z_min)/(z_max-z_min)
    airfoil = np.zeros_like(airfoil0)
                
    for i in range(m):
        for j in range(n):
            airfoil += bernstein(xx, m-1, i).reshape(-1,1) * \
                        bernstein(zz, n-1, j).reshape(-1,1) * P[j,i].reshape(1,2)
            
    return airfoil
    

if __name__ == '__main__':
    
    m = 4
    n = 3
    
    initial_path = '../initial_airfoil/naca0012.dat'
    airfoil0_true = np.loadtxt(initial_path, skiprows=1)
    
    x_min = np.min(airfoil0_true[:,0])
    x_max = np.max(airfoil0_true[:,0])
    z_min = np.min(airfoil0_true[:,1])
    z_max = np.max(airfoil0_true[:,1])
    Px = np.linspace(x_min, x_max, m, endpoint=True)
    Py = np.linspace(z_min, z_max, n, endpoint=True)
    x, y = np.meshgrid(Px, Py)
    P0 = np.stack((x, y), axis=-1)
    Px = P0[:,:,0]
    alpha0 = P0[:,:,1].flatten()
    
    airfoil0 = synthesize(alpha0, airfoil0_true, m, n, Px)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(airfoil0[:,0], airfoil0[:,1], 'o-')
    plt.plot(airfoil0_true[:,0], airfoil0_true[:,1], 'r-')
    plt.plot(P0[:,:,0].flatten(), P0[:,:,1].flatten(), 'rs')
    plt.axis('equal')
    plt.show()


    

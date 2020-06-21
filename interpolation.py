"""
B-spline approximation.

Author(s): Wei Chen (wchen459@umd.edu)

Reference(s): 
    [1] Lepine, Jerome, Guibault, Francois, Trepanier, Jean-Yves, Pepin, Francois. (2001). 
        Optimized nonuniform rational B-spline geometrical representation for aerodynamic 
        design of wings. AIAA journal, 39(11), 2033-2041.
    [2] Lepine, J., Trepanier, J. Y., & Pepin, F. (2000, January). Wing aerodynamic design 
        using an optimized NURBS geometrical representation. In 38th Aerospace Sciences 
        Meeting and Exhibit (p. 669).

n+1 : number of control points
m+1 : number of data points
"""

import numpy as np
from scipy.interpolate import splev, splprep, interp1d
from scipy.integrate import cumtrapz


def interpolate(Q, N, k, D=20, resolution=1000):
    ''' Interpolate N points whose concentration is based on curvature. '''
    res, fp, ier, msg = splprep(Q.T, u=None, k=k, s=1e-6, per=0, full_output=1)
    tck, u = res
    uu = np.linspace(u.min(), u.max(), resolution)
    x, y = splev(uu, tck, der=0)
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    cv = np.abs(ddx*dy - dx*ddy)/(dx*dx + dy*dy)**1.5 + D
    cv_int = cumtrapz(cv, uu, initial=0)
    fcv = interp1d(cv_int, uu)
    cv_int_samples = np.linspace(0, cv_int.max(), N)
    u_new = fcv(cv_int_samples)
    x_new, y_new = splev(u_new, tck, der=0)
    xy_new = np.vstack((x_new, y_new)).T
    return xy_new


if __name__ == "__main__":
    
    import os
    from initial_airfoil.cartesian import read_cartesian
    from matplotlib import pyplot as plt
    
    N = 192
    k = 3
#    data_path = './initial_airfoil/naca0012.dat'
#    name = os.path.splitext(os.path.basename(data_path))[0]
#    
#    Q = read_cartesian(data_path)
#    xy_new = interpolate(Q, N, k)
#    
#    np.savetxt('./initial_airfoil/naca0012_{}.dat'.format(N), xy_new, delimiter=',', fmt='%1.6f')
#    
#    plt.figure()
#    plt.plot(Q[:,0], Q[:,1], 'ro', alpha=.5)
#    plt.plot(xy_new[:,0], xy_new[:,1], 'bo-', alpha=.5)
#    plt.axis('equal')
#    plt.xlim(-0.1, 1.1)
#    plt.show()
    
    D = 50
    
    data_path = './data/airfoil_interp.npy'
    airfoils = np.load(data_path)
    new_airfoils = []
    for airfoil in airfoils:
        new_airfoil = interpolate(airfoil, N, k, D)
        new_airfoils.append(new_airfoil)
    np.save('./data/airfoil_interp_uniform.npy', new_airfoils)
    
    data_path = './initial_airfoil/naca0012.dat'
    Q = read_cartesian(data_path)
    xy_new = interpolate(Q, N, k, D)
    if xy_new[N//2,1]<0:
        pre_new = xy_new[-N//2:]
        suc_new = np.vstack((pre_new[:,0], -pre_new[:,1])).T
        suc_new = np.flip(suc_new, axis=0)
    else:
        suc_new = xy_new[:N//2]
        pre_new = np.vstack((suc_new[:,0], -suc_new[:,1])).T
        pre_new = np.flip(pre_new, axix=0)
    xy_new = np.vstack((suc_new, pre_new))
    np.savetxt('./initial_airfoil/naca0012_uniform_{}.dat'.format(N), xy_new, delimiter=',', fmt='%1.6f')
    
    plt.figure()
    plt.plot(xy_new[:,0], xy_new[:,1], 'bo-', ms=3, alpha=.5)
    plt.axis('equal')
    plt.xlim(-0.1, 1.1)
    plt.show()

    
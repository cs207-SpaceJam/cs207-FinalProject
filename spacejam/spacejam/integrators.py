import numpy as np
from . import dual, autodiff

""" Suite of implicit integrators. These methods include the first three
Adams-Moulton orders (i.e. s = 0, 1, 2).
See: https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Moulton_methods

"""

def amso(func, X_old, h=1E-3, X_tol=1E-1, i_tol=1E2, kwargs=None):
    """ (s=0) Adams-Moulton method

    Parameters
    ----------
    yea: yea yea
    """
    if kwargs:
        ad = autodiff.AutoDiff(func, X_old, kwargs=kwargs)
    else:
        ad = autodiff.AutoDiff(func, X_old)
    
    # Initial guess with forward Euler
    X_new = X_old + h*ad.r.flatten() # X_{n+1}^(0)
    
    # Iterate to better solution for X_new using Newton-Raphson method
    # on backward Euler implementation
    X_iold = X_old
    X_inew = X_new
    i = 0
    while np.linalg.norm(X_inew - X_iold) > X_tol:
        X_iold = X_inew

        if i > i_tol:
            # print('solution did not converge')
            return None

        ad = autodiff.AutoDiff(func, X_iold, kwargs=kwargs) # get Jacobian (ad.d)
        I = np.eye(len(ad.r.flatten()))
        D = I - h*ad.d
        g = X_iold - X_old - h*ad.r.flatten()

        X_inew = X_iold - np.dot(np.linalg.pinv(D), g)

        # update
        i += 1

    # update
    X_new = X_iold
    return X_new

def amsi(func, X_old, h=1E-3, X_tol=1E-1, i_tol=1E2, kwargs=None):
    if kwargs:
        ad = autodiff.AutoDiff(func, X_old, kwargs=kwargs)
    else:
        ad = autodiff.AutoDiff(func, X_old)
    
    # Initial guess with forward Euler
    X_new = X_old + h*ad.r.flatten() # X_{n+1}^(0)
    
    # Iterate to better solution for X_new using Newton-Raphson method
    # on backward Euler implementation
    X_iold = X_old
    X_inew = X_new
    i = 0
    while np.linalg.norm(X_inew - X_iold) > X_tol:
        X_iold = X_inew

        if i > i_tol:
            sys.exit('solution did not converge')

        ad = autodiff.AutoDiff(func, X_iold, kwargs=kwargs) # get Jacobian (ad.d)
        ad_n = autodiff.AutoDiff(func, X_old, kwargs=kwargs) # for X_n
        I = np.eye(len(ad.r.flatten()))
        D = I - (h/2)*ad.d
        g = X_iold - X_old - (h/2)*ad.r.flatten() - (h/2)*ad_n.r.flatten()

        X_inew = X_iold - np.dot(np.linalg.pinv(D), g)

        # update
        i += 1

    # update
    X_new = X_iold
    return X_new

def amsii(func, X_n, X_nn, h=1E-3, X_tol=1E-1, i_tol=1E2, kwargs=None):
    if kwargs:
        ad = autodiff.AutoDiff(func, X_n, kwargs=kwargs)
    else:
        ad = autodiff.AutoDiff(func, X_n)
    
    # Initial guess with forward Euler
    X_new = X_n + h*ad.r.flatten() # X_{n+1}^(0)
    
    # Iterate to better solution for X_new using Newton-Raphson method
    # on backward Euler implementation
    X_iold = X_n
    X_inew = X_new
    i = 0
    while np.linalg.norm(X_inew - X_iold) > X_tol:
        X_iold = X_inew

        if i > i_tol:
            sys.exit('solution did not converge')

        ad = autodiff.AutoDiff(func, X_iold, kwargs=kwargs) # get Jacobian (ad.d)
        ad_n = autodiff.AutoDiff(func, X_n, kwargs=kwargs) # for X_n
        ad_nn = autodiff.AutoDiff(func, X_nn, kwargs=kwargs) # for X_n-1
        I = np.eye(len(ad.r.flatten()))
        D = I - (5*h/12)*ad.d
        g1 = X_iold
        g2 = -X_n
        g3 = -(5*h/12)*ad.r.flatten()
        g4 = -(2*h/3)*ad_n.r.flatten()
        g5 = -(h/12)*ad_nn.r.flatten()
        g = g1 + g2 + g3 + g4 + g5
        X_inew = X_iold - np.dot(np.linalg.pinv(D), g)
        # update
        i += 1

    # update
    X_new = X_iold
    return X_new

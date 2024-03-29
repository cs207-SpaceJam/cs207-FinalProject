import numpy as np
import sys
from . import dual, autodiff

""" Suite of implicit integrators. These methods include the first three
Adams-Moulton orders (i.e. s = 0, 1, 2).
See: https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Moulton_methods
"""


def amso(func, X_old, h=1e-3, X_tol=1e-1, i_tol=1e2, kwargs=None):
    """Zeroth order Adams-Moulton method (AKA Backward Euler)

    Parameters
    ----------
    func : function
        User defined function to be integrated.
    X_old : numpy.ndarray
            Initial input to user function
    h : float (default 1E-3)
        Timestep
    X_tol : float (default 1E-1)
            Minimum difference between Newton-Raphson iterates to terminate on.
    i_tol : int (default 1E2)
            Maximum number of Newton-Raphson iterations. Entire simulation
            terminates if this number is exceeded.
    kwargs : dict (default None)
             optional arguments to be supplied to user defined function.

    Returns
    -------
    X_new : numpy.ndarray
            Final X_n+1 found from root finding of implicit method

    Examples
    --------
    >>>
    """
    if kwargs:
        ad = autodiff.AutoDiff(func, X_old, kwargs=kwargs)
    else:
        ad = autodiff.AutoDiff(func, X_old)

    # Initial guess with forward Euler
    X_new = X_old + h * ad.r.flatten()  # X_{n+1}^(0)

    # Iterate to better solution for X_new using Newton-Raphson method
    X_iold = X_old
    X_inew = X_new
    i = 0
    while np.linalg.norm(X_inew - X_iold) > X_tol:
        X_iold = X_inew

        if i > i_tol:
            msg = (
                "\nSorry, spacejam did not converge for s=0 A-M method.\n"
                "Try adjusting X_tol, i_tol, or using another integrator."
            )
            sys.exit(msg)

        ad = autodiff.AutoDiff(func, X_iold, kwargs=kwargs)  # get Jacobian (ad.d)
        I = np.eye(len(ad.r.flatten()))
        D = I - h * ad.d
        g = X_iold - X_old - h * ad.r.flatten()

        X_inew = X_iold - np.dot(np.linalg.pinv(D), g)

        # update
        i += 1

    # update
    X_new = X_iold
    return X_new


def amsi(func, X_old, h=1e-3, X_tol=1e-1, i_tol=1e2, kwargs=None):
    """First order Adams-Moulton method (AKA Trapezoid)

    Parameters
    ----------
    func : function
        User defined function to be integrated.
    X_old : numpy.ndarray
            Initial input to user function
    h : float (default 1E-3)
        Timestep
    X_tol : float (default 1E-1)
            Minimum difference between Newton-Raphson iterates to terminate on.
    i_tol : int (default 1E2)
            Maximum number of Newton-Raphson iterations. Entire simulation
            terminates if this number is exceeded.
    kwargs : dict (default None)
             optional arguments to be supplied to user defined function.

    Returns
    -------
    X_new : numpy.ndarray
            Final X_n+1 found from root finding of implicit method
    """
    if kwargs:
        ad = autodiff.AutoDiff(func, X_old, kwargs=kwargs)
    else:
        ad = autodiff.AutoDiff(func, X_old)

    # Initial guess with forward Euler
    X_new = X_old + h * ad.r.flatten()  # X_{n+1}^(0)

    # Iterate to better solution for X_new using Newton-Raphson method
    X_iold = X_old
    X_inew = X_new
    i = 0
    while np.linalg.norm(X_inew - X_iold) > X_tol:
        X_iold = X_inew

        if i > i_tol:
            msg = (
                "\nSorry, spacejam did not converge for s=1 A-M method.\n"
                "Try adjusting X_tol, i_tol, or using another integrator."
            )
            sys.exit(msg)

        ad = autodiff.AutoDiff(func, X_iold, kwargs=kwargs)  # get Jacobian (ad.d)
        ad_n = autodiff.AutoDiff(func, X_old, kwargs=kwargs)  # for X_n
        I = np.eye(len(ad.r.flatten()))
        D = I - (h / 2) * ad.d
        g = X_iold - X_old - (h / 2) * ad.r.flatten() - (h / 2) * ad_n.r.flatten()

        X_inew = X_iold - np.dot(np.linalg.pinv(D), g)

        # update
        i += 1

    # update
    X_new = X_iold
    return X_new


def amsii(func, X_n, X_nn, h=1e-3, X_tol=1e-1, i_tol=1e2, kwargs=None):
    """Second order Adams-Moulton method

    Parameters
    ----------
    func : function
        User defined function to be integrated.
    X_n : numpy.ndarray
          X_n
    X_nn : numpy.ndarray
           X_n-1
    h : float (default 1E-3)
        Timestep
    X_tol : float (default 1E-1)
            Minimum difference between Newton-Raphson iterates to terminate on.
    i_tol : int (default 1E2)
            Maximum number of Newton-Raphson iterations. Entire simulation
            terminates if this number is exceeded.
    kwargs : dict (default None)
             optional arguments to be supplied to user defined function.

    Returns
    -------
    X_new : numpy.ndarray
            Final X_n+1 found from root finding of implicit method
    """
    if kwargs:
        ad = autodiff.AutoDiff(func, X_n, kwargs=kwargs)
    else:
        ad = autodiff.AutoDiff(func, X_n)

    # Initial guess with forward Euler
    X_new = X_n + h * ad.r.flatten()  # X_{n+1}^(0)

    # Iterate to better solution for X_new using Newton-Raphson method
    X_iold = X_n
    X_inew = X_new
    i = 0
    while np.linalg.norm(X_inew - X_iold) > X_tol:
        X_iold = X_inew

        if i > i_tol:
            msg = (
                "\nSorry, spacejam did not converge for s=2 A-M method.\n"
                "Try adjusting X_tol, i_tol, or using another integrator."
            )
            sys.exit(msg)

        ad = autodiff.AutoDiff(func, X_iold, kwargs=kwargs)  # get Jacobian (ad.d)
        ad_n = autodiff.AutoDiff(func, X_n, kwargs=kwargs)  # X_n
        ad_nn = autodiff.AutoDiff(func, X_nn, kwargs=kwargs)  # X_n-1
        I = np.eye(len(ad.r.flatten()))
        D = I - (5 * h / 12) * ad.d
        g1 = X_iold
        g2 = X_n
        g3 = (5 / 12) * ad.r.flatten()
        g4 = (2 / 3) * ad_n.r.flatten()
        g5 = (1 / 12) * ad_nn.r.flatten()
        g = g1 - g2 - h * (g3 + g4 - g5)
        X_inew = X_iold - np.dot(np.linalg.pinv(D), g)

        # update
        i += 1

    # update
    X_new = X_iold
    return X_new

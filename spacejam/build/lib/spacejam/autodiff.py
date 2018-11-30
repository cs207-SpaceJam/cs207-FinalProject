import numpy as np
from . import dual

class AutoDiff():
    """ Performs automatic differentiation (AD) on functions input by user.

    AD if performed by transforming `f(x1, x2, ...)` to `f(p_x1, p_x2, ...)`,
    where `p_xi` is returned from :any:`spacejam.dual.Dual` . 
    The final result is then returned in a series of 1D `numpy.ndarray` or 
    formatted matrices depending on if the user specified functions F are
    multivariable or not.  

    Attributes
    ----------
    r : numpy.ndarray
        User defined function(s) `F` evaluated at `p`.
    d : numpy.ndarray
        Corresponding derivative, gradient, or Jacobian of user defined
        functions(s).
    """

    def __init__(self, func, p, kwargs=None):
        """ 
        Parameters
        ----------
        func : numpy.ndarray
            user defined function(s).
        p : numpy.ndarray
            user defined point(s) to evaluate derivative/gradient/Jacobian at.
        """
        if kwargs:
            result = self._ad(func, p, kwargs=kwargs) # perform AD
        else:
            result = self._ad(func, p) # perform AD

        # returns func(p) and approriate J func(p) or grad func(p)
        # in real and dual part of AutoDiff class, respectively
        if len(result) == 1: # scalar F, scalar or vector p
            # hacky way to get numpy formatting to work
            if len(p) == 1: self.r = result[0].r
            else: self.r = np.array([result[0].r])
            # load dual part
            self.d = result[0].d
            # load full func(p) + epsilon f'(p) or
            # func(p) + epsilon grad f(p)
            self._full = result[0]

        else: # vector F, scalar or vector p
            # format as F(p) column vector and Jacobian matrix
            # for real and dual part, respectively
            self.r, self.d = self._matrix(func, p, result)
            self._full = result

    def _ad(self, func, p, kwargs=None):
        """ Internally computes `func(p)` and its derivative(s).
        
        Notes
        -----
        `_ad` returns a nested 1D `numpy.ndarray` to be formatted internally
        accordingly in :any:`spacejam.autodiff.AutoDiff.__init__`  .

        Parameters
        ----------
        func : numpy.ndarray
            function(s) specified by user.
        p : numpy.ndarray
            point(s) specified by user.
        """
        if len(p) == 1: # scalar p
            p_mult = np.array([dual.Dual(p)]) 

        else:# vector p
            p_mult = [dual.Dual(pi, idx=i, x=p) for i, pi in enumerate(p)]
            p_mult = np.array(p_mult) # convert list to numpy array

        # perform AD with specified function(s)
        if kwargs:
            result = func(*p_mult, **kwargs) 
        else:
            result = func(*p_mult) 
        return result

    def _matrix(self, F, p, result): 
        """ Internally formats `result` returned by 
        :any:`spacejam.autodiff.AutoDiff._ad` into matrices.
        
        Parameters
        ----------
        F : numpy.ndarray
            functionss specified by user.
        p : numpy.ndarray
            point(s) specified by user.
        result: numpy.ndarray
            Nested 1D numpy.ndarray to be formatted into matrices.
        
        Returns
        -------
        Fs : numpy.ndarray
            Column matrix of functions evaluated at points(s).
        jac : numpy.ndarray 
            Corresponding Jacobian matrix.
        """
        # returns formatted F(p) and Jacobian matrices 
        Fs = np.empty((result.size, 1)) # initialze empty F(p)
        jac = np.empty((result.size, p.size)) # initialize empty J F(p)
        for i, f in enumerate(result): # fill in each row of each
            Fs[i] = f.r
            jac[i] = f.d
        return Fs, jac

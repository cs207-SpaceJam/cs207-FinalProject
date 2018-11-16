import numpy as np
from . import dual as du

class AutoDiff():
    def __init__(self, func, p):
        result = self._ad(func, p) # perform AD

        # returns func(p) and approriate J func(p) or grad func(p)
        # in real and dual part of AutoDiff class, respectively
        if len(func(*p)) == 1: # scalar F, scalar or vector p
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
            self.r, self.d = self._jac(func, p, result)
            self._full = result

    def _ad(self, func, p):
        if len(p) == 1: # scalar p
            p_mult = np.array([du.Dual(p)]) 

        else:# vector p
            p_mult = [du.Dual(pi, idx=i, x=p) for i, pi in enumerate(p)]
            p_mult = np.array(p_mult) # convert list to numpy array

        result = func(*p_mult) # perform AD with specified function(s)
        return result

    def _jac(self, F, p, result): # returns formatted Jacobian matrix 
        evals = np.empty((F(*p).size, 1)) # initialze empty F(p)
        jac = np.empty((F(*p).size, p.size)) # initialize empty J F(p)
        for i, f in enumerate(result): # fill in each row of each
            evals[i] = f.r
            jac[i] = f.d
        return evals, jac

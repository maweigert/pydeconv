"""


mweigert@mpi-cbg.de

"""
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp2d

class ThresholdingLookup(object):

    def __init__(self,alpha = 2., v_max = 10., n_v = 256, b_max = 10., n_b = 128):
        self.a = alpha
        self.v_arr = np.linspace(0,v_max,n_v)
        self.b_arr = np.sqrt(2)**np.linspace(0,2.*np.log2(b_max),n_b)
        self._assemble()

    def _prox(self,b):
        def _func(_x):
            f = lambda w: abs(w)**self.a+.5*b*(w-_x)**2
            return minimize_scalar(f,(0.1*_x,_x)).x
        return _func


    def _get_w_for_b(self,b):
        """
        w(v) = argmin_w |w|^a+b/2*(w-v)^2

        which we solve by deriving dw/dv and then integrating it
        """
        
        if self.a ==2:
            w_star = 0.
        else:
            w_star = ((1-self.a)*self.a/b)**(1./(2.-self.a))

        def grad_w(w,v):
            if w>w_star:
                return 1.*b/(b+self.a*(self.a-1)*w**(self.a-2.))
            else:
                return -1.

        return odeint(grad_w,self._prox(b)(self.v_arr[-1]),self.v_arr[::-1]).flatten()[::-1]

    def _assemble(self, method = "use_min"):
        print("calculating lookup table")

        if method == "use_grad":
            self.grid = np.zeros((len(self.b_arr),len(self.v_arr)))
            for i,b in enumerate(self.b_arr):
                self.grid[i] = self._get_w_for_b(b)
        elif method == "use_min":
            self.grid = np.array([[self._prox(b)(v) for b in self.b_arr] for v in self.v_arr])
        else:
            raise NotImplementedError

        self._interp = interp2d(self.b_arr, self.v_arr,self.grid, bounds_error=True)

    def __call__(self,v,b, is_sorted = False):

        assert np.isscalar(b)

        if np.isscalar(v):
            return np.sign(v)*self._interp(b,np.abs(v))

        v_abs = np.abs(v.flatten())


        # make it work for scalar and array v
        res = self._interp(b,v_abs).flatten()

        if not is_sorted:
            #restore original order, as scipy.interp2d sorts them implicitly
            rev_order = np.empty(len(v_abs),np.int)
            rev_order[np.argsort(v_abs)] = np.arange(len(v_abs))

            res = res[rev_order]

        return np.sign(v)*res.reshape(v.shape)



if __name__ == '__main__':

    look = ThresholdingLookup(alpha = .3,v_max = 10., b_max= 4.)

    # V, B = np.meshgrid(look.v_arr,look.b_arr, indexing="ij")
    #
    # def soft_thresh(x,lam):
    #     return np.sign(x)*np.maximum(0.,np.abs(x)-lam)
    #
    # grid2 = soft_thresh(V,1./B)
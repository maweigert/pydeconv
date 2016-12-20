""" a small wrapper around fftw objects for constant input shape"""



import numpy as np

import pyfftw
import pyfftw.interfaces.numpy_fft as fftw

import os
import pickle


pyfftw.interfaces.cache.enable()



class MyFFTW(object):
    """ a simple wrapper around fftw
    """
    
    flags = ('FFTW_MEASURE','FFTW_DESTROY_INPUT')
    _WISDOM_FILE = os.path.expanduser("~/.myfftw/wisdom")

    _complex_type = np.complex64
    _real_type = np.float32

    
    def __init__(self,shape, n_threads = 4, unitary = False):
        if not np.all([s%16==0 for s in shape]):
            raise ValueError("shape should be divisible by 16!")

        ax = range(len(shape))

        # the shape for the rfftn
        rshape = shape[:-1]+(shape[-1]/2+1,)

        # self.input_c = pyfftw.n_byte_align_empty(shape, 16, 'complex64')
        # self.input_r = pyfftw.n_byte_align_empty(shape, 16, 'float32')
        # self.input_c2 = pyfftw.n_byte_align_empty(rshape, 16, 'complex64')

        self.input_c = pyfftw.empty_aligned(shape,  'complex64')
        self.input_r = pyfftw.empty_aligned(shape,  'float32')
        self.input_c2 = pyfftw.empty_aligned(rshape, 'complex64')


        if unitary:
            self.prefac = 1.*np.sqrt(np.prod(shape))
        else:
            self.prefac  = None

        self.import_wisdom()

        kwargs = {"planner_effort":'FFTW_MEASURE',
                  "overwrite_input":True,
                  "auto_align_input":True,
                  "threads":n_threads}

        kwargs_r = {"planner_effort":'FFTW_MEASURE',
                  "threads":n_threads}
                  
        self._fftn = pyfftw.builders.fftn(self.input_c,**kwargs)
        self._ifftn = pyfftw.builders.ifftn(self.input_c,**kwargs)
        self._rfftn = pyfftw.builders.rfftn(self.input_r,**kwargs)
        self._irfftn = pyfftw.builders.irfftn(self.input_c2,**kwargs_r)

        
        self._zfftn = pyfftw.builders.fftn(self.input_c,axes= ax[1:],**kwargs)
        self._zifftn = pyfftw.builders.ifftn(self.input_c,axes= ax[1:],**kwargs)

        
        self.export_wisdom()

    def fftn(self,x):
        # if not x.dtype.type is self._complex_type:
        #     raise ValueError("wrong input type! Should be %s but is %s"%(self._complex_type, x.dtype.type))


        self.input_c[:] = x.astype(self._complex_type, copy = False)

        res = self._fftn().copy()

        if self.prefac:
            res /= self.prefac
        return res
        
    def ifftn(self,x):
        # if not x.dtype.type is self._complex_type:
        #     raise ValueError("wrong input type! Should be %s but is %s"%(self._complex_type, x.dtype.type))

        self.input_c[:] = x.astype(self._complex_type, copy = False)

        res = self._ifftn().copy()
        if self.prefac:
            res *= self.prefac
        return res


    def rfftn(self,x):
        # if not x.dtype.type is self._real_type:
        #     raise ValueError("wrong input type! Should be %s but is %s"%(self._real_type, x.dtype.type))

        self.input_r[:] = x.astype(self._real_type, copy = False)
        res = 1.*self._rfftn().copy()
        if self.prefac:
            res /= self.prefac
        return res


    def irfftn(self,x):
        # if not x.dtype.type is self._complex_type:
        #     raise ValueError("wrong input type! Should be %s but is %s"%(self._complex_type, x.dtype.type))

        self.input_c2[:] = x.astype(self._complex_type, copy = False)
        res = self._irfftn().copy()
        if self.prefac:
            res *= self.prefac
        return res

        
    def zfftn(self,x):
        """ dont transform along first dimension"""
        # if not x.dtype.type is self._complex_type:
        #     raise ValueError("wrong input type! Should be %s but is %s"%(self._complex_type, x.dtype.type))

        self.input_c[:] = x.astype(self._complex_type, copy = False)
        res = self._zfftn().copy()
        if self.prefac:
            res /= self.prefac
        return res


    def zifftn(self,x):
        """ dont transform along first dimension"""
        # if not x.dtype.type is self._complex_type:
        #     raise ValueError("wrong input type! Should be %s but is %s"%(self._complex_type, x.dtype.type))

        self.input_c[:] = x.astype(self._complex_type, copy = False)
        res = self._zifftn().copy()
        if self.prefac:
            res *= self.prefac
        return res

        
    def export_wisdom(self):
        wis = pyfftw.export_wisdom()

        basedir = os.path.dirname(MyFFTW._WISDOM_FILE)
        
        if not os.path.exists(basedir):
            os.makedirs(basedir)
            
        with open(MyFFTW._WISDOM_FILE,"w") as f:
            pickle.dump(wis,f)

    def import_wisdom(self):
        try:
            wis = pickle.load(open(MyFFTW._WISDOM_FILE,"r"))
            pyfftw.import_wisdom(wis)
        except Exception as e :
            print e
        
        
if __name__ == '__main__':
    from time import time

    # sogme testing

    N = 256
    
    x = np.ones((N,)*3)

    f = MyFFTW(x.shape, n_threads = 4)

    t = time()
    a = f.rfftn(x)
    b = f.irfftn(a)

    u = f.fftn(x)
    v = f.ifftn(u)

    c = f.zfftn(x)

    print time()-t
    
    

    
    

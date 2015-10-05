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
    
    def __init__(self,shape, n_threads = 4):
        if not np.all([s%16==0 for s in shape]):
            raise ValueError("shape should be divisble by 16!")

        ax = range(len(shape))

        # the shape for the rfftn
        rshape = shape[:-1]+(shape[-1]/2+1,)

        self.input_c = pyfftw.n_byte_align_empty(shape, 16, 'complex64')

        self.input_r = pyfftw.n_byte_align_empty(shape, 16, 'float32')

        self.input_c2 = pyfftw.n_byte_align_empty(rshape, 16, 'complex64')



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
        self.input_c[:] = x
        return self._fftn().copy()
        
    def ifftn(self,x):
        self.input_c[:] = x
        return self._ifftn().copy()

    def rfftn(self,x):
        self.input_r[:] = x
        return 1.*self._rfftn().copy()

    def irfftn(self,x):
        self.input_c2[:] = x
        return self._irfftn().copy()
        
    def zfftn(self,x):
        """ dont transform along first dimension"""
        self.input_c[:] = x
        return self._zfftn().copy()

    def zifftn(self,x):
        """ dont transform along first dimension"""
        self.input_c[:] = x
        return self._zifftn().copy()
        
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
    
    

    
    

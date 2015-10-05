""" joint wiener deconvolution """

import numpy as  np
                
from pydeconv._fftw.myfftw import MyFFTW


def deconv_wiener(ys, hs,
                   gamma = 1.e-6,
                   n_threads = 6,
                   y_is_fft = False,
                   h_is_fft = False):
    
    """ wiener deconv

    y is/are the recorded image(s)
    h is/are the kernel(s)

    both can be equal sized tuples, in which case a joint deconvolution is performed

    if the data or the kernels are already fourier transformed then
    set y_is_fft/h_is_fft to True
    
    """

    if not isinstance(ys,(list,tuple)):
        ys = [ys]

    if not isinstance(hs,(list,tuple)):
        hs = [hs]
        
    if len(ys) != len(hs):
        raise ValueError("len(y) != len(h)   %d != %d"%(len(ys),len(hs)))

    if not np.all([_y.shape == _h.shape for _y, _h in zip(ys,hs)]):
        raise ValueError("y and h have non compatible shapes...")

    if not y_is_fft:
        dshape = ys[0].shape
    else:
        dshape = ys[0].shape[:-1]+(2*(ys[0].shape[-1]-1),)
    
    FFTW = MyFFTW(dshape,n_threads = n_threads)

    if not h_is_fft:
        Hs = [FFTW.rfftn(h) for h in hs]
    else:
        Hs = hs
        
    if not y_is_fft:
        Ys = [FFTW.rfftn(y) for y in ys]
    else:
        Ys = ys
        
    U = reduce(np.add,[_H.conjugate()*_Y for _H,_Y in zip(Hs,Ys)])
    
    U /= 1.*gamma+reduce(np.add,[np.abs(_H)**2 for _H in Hs])
        
    return FFTW.irfftn(U)

if __name__ == '__main__':

    pass

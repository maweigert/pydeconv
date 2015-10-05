
import numpy as np

def myconvolve(x,h, is_fft = False):
    if is_fft:
        x_f = x
        h_f = h
    else:
        x_f = np.fft.rfftn(x)
        h_f = np.fft.rfftn(h)
    
    return np.abs(np.fft.irfftn(x_f*h_f))

def psf(dshape,sigmas = (2.,2.)):
    Xs = np.meshgrid(*[np.arange(-_s/2,_s/2) for _s in dshape], indexing="ij")

    h = np.exp(-np.sum([_X**2/2./_s**2 for _X,_s in zip(Xs,sigmas)],axis=0))

    h *= 1./np.sum(h)
    return np.fft.ifftshift(h)
    

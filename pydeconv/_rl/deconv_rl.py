""" joint wiener deconvolution """

import numpy as  np
                
from pydeconv._fftw.myfftw import MyFFTW


def deconv_rl(ys, hs,
              Niter=10,
              gamma = 1.e-2,
              n_threads = 6,
              y_is_fft = False,
              h_is_fft = False,
              log_iter = False):
    
    """ richardson lucy deconvolution

    y is/are the recorded image(s)
    h is/are the kernel(s)

    both can be equal sized tuples, in which case a joint deconvolution is performed

    if the data or the kernels are already fourier transformed then
    set y_is_fft/h_is_fft to True

    increase gamma if something explodes....
    
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

    if log_iter:
        print "creating FFTW object"

    FFTW = MyFFTW(dshape,n_threads = n_threads)

    if ys[0].ndim==1:
        hs_flip = [h[::-1] for h in hs]
    elif ys[0].ndim==2:
        hs_flip = [h[::-1,::-1] for h in hs]
    elif ys[0].ndim==3:
        hs_flip = [h[::-1,::-1,::-1] for h in hs]
    else:
        raise NotImplementedError("data dimension %s not supported"%ys[0].ndim)

    if log_iter:
        print "setting up ffts"

    if not h_is_fft:
        Hs = [FFTW.rfftn(h) for h in hs]
        Hs_flip = [FFTW.rfftn(h) for h in hs_flip]
    else:
        Hs = hs
        Hs_flip = hs_flip

        
    if not y_is_fft:
        Ys = [FFTW.rfftn(y) for y in ys]
    else:
        Ys = ys

    def _single_lucy_step(d,u_f,h_f,h_flip_f):
        tmp = FFTW.irfftn(u_f*h_f)
        tmp2 = FFTW.rfftn(d/(tmp+gamma))
        tmp3 = FFTW.irfftn(tmp2*h_flip_f)
        return tmp3



    u = np.mean(ys[0])*np.ones_like(ys[0])

    for i in range(Niter):
        if log_iter:
            print "deconv_rl step %s/%s"%(i+1,Niter)
        U = FFTW.rfftn(u)
        us = [_single_lucy_step(y,U,H,H_flip) for y,H,H_flip in zip(ys,Hs,Hs_flip)]

        u *= reduce(np.multiply,us)
        #return u,us[0]

    return u


if __name__ == '__main__':

    from matplotlib.pyplot import imread
    from pydeconv.utils import myconvolve, psf

    im = imread("../tests2/data/usaf.png")


    np.random.seed(0)


    hx = (5.,5.)
    h = psf(im.shape,hx)

    h = np.roll(np.roll(h,20,axis=0),10,axis=1)

    g = myconvolve(im ,h)

    g += .01*np.amax(im)*np.random.normal(0,1.,im.shape)


    u = deconv_rl(g,h,20, gamma = 0.1)
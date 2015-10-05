""" tv deconvolution 2d or 2d version

see http://www.ccom.ucsd.edu/~peg/papers/ALvideopaper.pdf

"""
import numpy as np

from pydeconv._fftw.myfftw import MyFFTW


def soft_thresh(x,t = 1.):
    return np.sign(x)*np.maximum(np.abs(x)-t,0)

def divergence(u):
    """the divergence of vector field u where u[i,...] are the components """
    return reduce(np.add,[np.gradient(u[i])[i] for i in range(len(u))])

    
def dft_lap(dshape,units = None, use_rfft = False):
    """ the dft of the laplacian stencil in 2d"""
    if units is None:
        units = (1.,)*len(dshape)

    kxs = [np.fft.fftfreq(_s,_u) for _s,_u in zip(dshape,units)]

    if use_rfft:
        kxs[-1] = kxs[-1][:dshape[-1]//2+1]

    KXs = np.meshgrid(*kxs,indexing="ij")
    
    h = np.sum([2*np.cos(2.*np.pi*_K) for _K in KXs],axis=0) - 2.*len(KXs)
    h *= 1./np.prod(units)
    return h


    

def _deconv_tv_al_fft(ys, hs,
                 mu = 1000.,
                 rho = 2., 
                 Niter = 10,
                 n_threads = 6,
                 tv_norm = "isotropic",
                 y_is_fft = False,
                 h_is_fft = False):
    
    """ total variation deconv

    y is/are the recorded image(s)
    h is/are the kernel(s)

    both can be equal sized tuples, in which case a joint deconvolution is performed

    if the data or the kernels are already fourier transformed then
    set y_is_fft/h_is_fft to True

    tv_norm = "isotropic","anisotropic"
    
    """

    if not isinstance(ys,(list,tuple)):
        ys = [ys]

    if not isinstance(hs,(list,tuple)):
        hs = [hs]
        
    if len(ys) != len(hs):
        raise ValueError("len(y) != len(h)   %d != %d"%(len(ys),len(hs)))

    if not np.all([_y.shape == _h.shape for _y, _h in zip(ys,hs)]):
        raise ValueError("y and h have non compatible shapes...")

    dshape = ys[0].shape
    
    # FFTW = MyFFTW(dshape,n_threads = n_threads)

    # if not h_is_fft:
    #     Hs = [FFTW.rfftn(h) for h in hs]

    # if not y_is_fft:
    #     Ys = [FFTW.rfftn(y) for y in ys]
    if not h_is_fft:
        Hs = [np.fft.fftn(h) for h in hs]

    if not y_is_fft:
        Ys = [np.fft.fftn(y) for y in ys]

    lap  = -dft_lap(dshape, use_rfft = False)
    
    f = 1.*ys[0]
    u = np.array(np.gradient(f))
    y = np.zeros((ys[0].ndim,)+dshape)
           

    rnorm = 0.
    for i in range(Niter):
        # print i, _fun(f,mu), rho

        # f subproblem

        f_f = 1.*mu/rho*reduce(np.add,[_H.conjugate()*_Y for _H,_Y in zip(Hs,Ys)])
        f_f -= np.fft.fftn(divergence(u-1./rho*y))

        f_f /= 1.*mu/rho*reduce(np.add,[np.abs(_H)**2 for _H in Hs]) + lap

        f = np.real(np.fft.ifftn(f_f))

        # u subproblem
        
        df = np.array(np.gradient(f))

        if tv_norm == "isotropic":
            # isotropic  case
            v = df + 1./rho*y
            mv = np.sqrt(np.sum(v**2,axis=0))
            mv[mv==0] = 1.
            tv = np.maximum(0,mv-1./rho)/mv
            u = tv*v
        else:
            # anisotropic case
            u = soft_thresh(df + 1./rho*y,1./rho)

        y -= rho*(u-df)

        print rho
        rnorm_new = np.sqrt(np.mean((u-df)**2))
        if rnorm_new > .7*rnorm:
            rho *= 2.
            
        rnorm = rnorm_new

    return f
    
def deconv_tv_al(ys, hs,
                 mu = 1000.,
                 rho = 2., 
                 Niter = 10,
                 n_threads = 6,
                 tv_norm = "isotropic",
                 y_is_fft = False,
                 h_is_fft = False):
    
    """ total variation deconv

    y is/are the recorded image(s)
    h is/are the kernel(s)

    both can be equal sized tuples, in which case a joint deconvolution is performed

    if the data or the kernels are already fourier transformed then
    set y_is_fft/h_is_fft to True

    tv_norm = "isotropic","anisotropic"
    
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

    lap  = -dft_lap(dshape, use_rfft = True)

    if not y_is_fft:
        f = 1.*ys[0]
    else:
        f = np.real(FFTW.irfftn(ys[0]))
    u = np.array(np.gradient(f))
    y = np.zeros((ys[0].ndim,)+dshape)
           

    rnorm = 0.
    for i in range(Niter):
        # print i, _fun(f,mu), rho

        # f subproblem

        f_f = 1.*mu/rho*reduce(np.add,[_H.conjugate()*_Y for _H,_Y in zip(Hs,Ys)])
        f_f -= FFTW.rfftn(divergence(u-1./rho*y))

        f_f /= 1.*mu/rho*reduce(np.add,[np.abs(_H)**2 for _H in Hs]) + lap

        f = np.real(FFTW.irfftn(f_f))

        # u subproblem
        
        df = np.array(np.gradient(f))

        if tv_norm == "isotropic":
            # isotropic  case
            v = df + 1./rho*y
            mv = np.sqrt(np.sum(v**2,axis=0))
            mv[mv==0] = 1.
            tv = np.maximum(0,mv-1./rho)/mv
            u = tv*v
        else:
            # anisotropic case
            u = soft_thresh(df + 1./rho*y,1./rho)

        y -= rho*(u-df)

        print rho
        rnorm_new = np.sqrt(np.mean((u-df)**2))
        if rnorm_new > .7*rnorm:
            rho *= 2.
            
        rnorm = rnorm_new

    return f
        
if __name__ == '__main__':
    from matplotlib.pyplot import imread
    from pydeconv.utils import myconvolve, psf
    
    im = imread("../tests/data/usaf.png")


    np.random.seed(0)

    
    hx = (5.,5.)
    h = psf(im.shape,hx)
        
    g = myconvolve(im ,h)
    
    g += .01*np.amax(im)*np.random.normal(0,1.,im.shape)

    mu = 10000.
    rho = 2.

    
    def wien(y_f,alpha = .1):
        u_f = h_f.conjugate()*y_f
        u_f /= np.abs(h_f)**2  + alpha
        return np.abs(np.fft.ifftn(u_f))

    def _fun(u , mu):
        gu = np.array(np.gradient(u))
        bu = myconvolve(u ,h)
        l1_part =  np.mean(reduce(np.add,[np.abs(_g) for _g in np.array(np.gradient(u))]))
        l2_part = np.mean((bu-y)**2)
        return l1_part + .5*mu*l2_part

    ys = g
    hs = h
    n_threads = 6
    

    
    if not isinstance(ys,(list,tuple)):
        ys = [ys]

    if not isinstance(hs,(list,tuple)):
        hs = [hs]
        
    if len(ys) != len(hs):
        raise ValueError("len(y) != len(h)   %d != %d"%(len(ys),len(hs)))

    if not np.all([_y.shape == _h.shape for _y, _h in zip(ys,hs)]):
        raise ValueError("y and h have non compatible shapes...")

    dshape = ys[0].shape
    
    
    Hs = [np.fft.fftn(h) for h in hs]

    Ys = [np.fft.fftn(y) for y in ys]

    lap  = -dft_lap(dshape)
    
    f = 1.*ys[0]
    u = np.array(np.gradient(f))
    y = np.zeros((ys[0].ndim,)+dshape)
           

    rnorm = 0.
    Niter = 15

    a = deconv_tv_al(ys,hs,mu,rho, Niter, tv_norm = "anisotropic")

    for i in range(Niter):
        # print i, _fun(f,mu), rho

        # f subproblem

        f_f = 1.*mu/rho*reduce(np.add,[_H.conjugate()*_Y for _H,_Y in zip(Hs,Ys)])
        f_f -= np.fft.fftn(divergence(u-1./rho*y))

        f_f /= 1.*mu/rho*reduce(np.add,[np.abs(_H)**2 for _H in Hs]) + lap

        f = np.real(np.fft.ifftn(f_f))

        # u subproblem
        
        df = np.array(np.gradient(f))

        # isotropic  case
        v = df + 1./rho*y
        mv = np.sqrt(np.sum(v**2,axis=0))
        mv[mv==0] = 1.
        tv = np.maximum(0,mv-1./rho)/mv
        u = tv*v

        # anisotropic case
        # u = soft_thresh(df + 1./rho*y,1./rho)

        y -= rho*(u-df)

        print rho
        rnorm_new = np.sqrt(np.mean((u-df)**2))
        if rnorm_new > .7*rnorm:
            rho *= 2.
            
        rnorm = rnorm_new


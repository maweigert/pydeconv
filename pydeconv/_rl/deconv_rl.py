""" joint wiener deconvolution """

from __future__ import absolute_import, division, print_function
import numpy as  np
                
from pydeconv._fftw.myfftw import MyFFTW
from six.moves import range
from six.moves import zip
from functools import reduce

def div_grad(u):
    grads = np.stack(np.gradient(u))
    normed = grads/(1.e-10+np.linalg.norm(grads,axis = 0))
    return reduce(np.add,[np.gradient(g,axis = i) for i,g in enumerate(normed)])


def deconv_rl(ys, hs,
              Niter=10,
              gamma = 1.e-5,
              n_threads = 6,
              h_is_fft = False,
              log_iter = False,
              mult_mode = "root",
              acceleration = 1.,
              tv_lambda = None,
              return_history = False,
              fft_is_unitary=False):

    """ richardson lucy deconvolution

    y is/are the recorded image(s)
    h is/are the kernel(s)

    both can be equal sized tuples, in which case a joint deconvolution is performed

    if the data or the kernels are already fourier transformed then
    set y_is_fft/h_is_fft to True

    increase gamma if something explodes....

    mult_mode = "root", "max", "min"

     definies how each lr channel is multiplied


    "root" and "min" should be preferred
    """



    if not isinstance(ys,(list,tuple)):
        ys = [ys]

    if not isinstance(hs,(list,tuple)):
        hs = [hs]
        
    if len(ys) != len(hs):
        raise ValueError("len(y) != len(h)   %d != %d"%(len(ys),len(hs)))


    if not h_is_fft and not np.all([_y.shape == _h.shape for _y, _h in zip(ys,hs)]):
        raise ValueError("y and h have non compatible shapes...")

    dshape = ys[0].shape


    if log_iter:
        print("creating FFTW object")

    FFTW = MyFFTW(dshape,n_threads = n_threads, unitary=fft_is_unitary)

    if not h_is_fft:
        if ys[0].ndim==1:
            hs_flip = [h[::-1] for h in hs]
        elif ys[0].ndim==2:
            hs_flip = [h[::-1,::-1] for h in hs]
        elif ys[0].ndim==3:
            hs_flip = [h[::-1,::-1,::-1] for h in hs]
        else:
            raise NotImplementedError("data dimension %s not supported"%ys[0].ndim)
    else:
        if ys[0].ndim==1:
            hs_flip = h
        elif ys[0].ndim==2:
            hs_flip = [h[::-1,:] for h in hs]
        elif ys[0].ndim==3:
            hs_flip = [h[::-1,::-1,:] for h in hs]
        else:
            raise NotImplementedError("data dimension %s not supported"%ys[0].ndim)


    if log_iter:
        print("setting up ffts")

    if not h_is_fft:
        Hs = [FFTW.rfftn(h) for h in hs]
        Hs_flip = [FFTW.rfftn(h) for h in hs_flip]
    else:
        Hs = hs
        Hs_flip = hs_flip

        

    def _single_lucy_multiplier(d,u_f,h_f,h_flip_f):
        tmp = np.abs(FFTW.irfftn(u_f*h_f))
        tmp2 = FFTW.rfftn(d/(tmp+gamma))
        tmp3 = np.abs(FFTW.irfftn(tmp2*h_flip_f))

        return tmp3


    u = np.mean([np.mean(_y) for _y in ys])*np.ones_like(ys[0])


    ums = [u.copy()]

    for i in range(Niter):
        if log_iter:
            print("deconv_rl step %s/%s"%(i+1,Niter))
        U = FFTW.rfftn(u)
        us = np.stack([_single_lucy_multiplier(y,U,H,H_flip) for y,H,H_flip in zip(ys,Hs,Hs_flip)])



        if mult_mode=="root":
            fac = reduce(np.multiply,us**(1./len(ys)))
        elif mult_mode =="prod":
            fac = reduce(np.multiply,us)
        elif mult_mode =="max":
            fac = np.amax(us,axis=0)
        elif mult_mode =="mean":
            fac = np.mean(us,axis=0)
        elif mult_mode =="min":
            fac = np.amin(us,axis=0)
        else:
            raise KeyError(multmode)

        fac = fac**acceleration

        if tv_lambda is None:
            u = u*fac
        else:
            u = u/(1.-tv_lambda*div_grad(u))*fac

        if return_history:
            ums.append(u.copy())


    if return_history:
        return ums
    else:
        return u


if __name__ == '__main__':

    import imgtools
    from pydeconv.utils import myconvolve, psf, psf_airy

    im = imgtools.test_images.usaf().astype(np.float32)

    im *= 1./255

    np.random.seed(0)
    #
    # h1 = psf_airy(im.shape, (10,20))
    #
    # h2 = psf_airy(im.shape, (20,10))
    # h3 = psf_airy(im.shape, (15,15))
    #
    # h4 = np.roll(psf_airy(im.shape, (10,10)),14,0)
    #
    # g1 = myconvolve(im ,h1)
    # g2 = myconvolve(im ,h2)
    # g3 = myconvolve(im ,h3)
    # g4 = myconvolve(im ,h4)
    #
    # g1 += .2*np.amax(im)*np.random.normal(0,1.,im.shape)
    # g2 += .2*np.amax(im)*np.random.normal(0,1.,im.shape)
    # g3 += .2*np.amax(im)*np.random.normal(0,1.,im.shape)
    # g4 += .1*np.amax(im)*np.random.normal(0,1.,im.shape)
    #


    #u = deconv_rl([g1,g2,g3],[h1,h2,h3],20, gamma = 0.000001)

    # a2 = deconv_rl([g1,g2],[h1,h2],10, gamma = 0.000001)
    # a3 = deconv_rl([g1,g2,g3],[h1,h2,h3],10, gamma = 0.000001)
    # a4 = deconv_rl([g1,g2,g3,g4],[h1,h2,h3,h4],10, gamma = 0.000001)
    #
    # u1 = deconv_rl([g1],[h1],20, gamma = 0.000001)
    # u2 = deconv_rl([g2],[h2],20, gamma = 0.000001)
    # u3 = deconv_rl([g3],[h3],20, gamma = 0.000001)
    # u4 = deconv_rl([g4],[h4],10, gamma = 0.000001)


    hs = [np.roll(psf_airy(im.shape, _d),2*i,0) for i,_d in enumerate([(10,20),(20,10),(15,15)])]
    gs = [myconvolve(im ,h)+.1*np.amax(im)*np.random.normal(0,1.,im.shape) for h in hs]


    u = deconv_rl([gs[0]]*3,[hs[0]]*3,20, gamma = 0.000001, fft_is_unitary=True)



    # from itertools import product
    # import pylab

    # N = len(gs)
    # u = []
    # for i,(g,h) in enumerate(zip(product(gs,repeat=2),product(hs,repeat=2))):
    #     u.append(deconv_rl(g,h,5,gamma = 1.e-5))
    #     pylab.subplot(N,N,i+1)
    #     pylab.imshow(u[-1])
    #     pylab.axis("off")
    #
    #
    # N = len(gs)
    # u = []
    # for i,(g,h) in enumerate(zip(product(gs,repeat=3),product(hs,repeat=3))):
    #     u.append(deconv_rl(g,h,10,gamma = 1.e-5, mult_mode="min"))
    #
    #     # pylab.subplot(N*np.ceil(np.sqrt(N)),N*np.ceil(np.sqrt(N)),i+1)
    #     # pylab.imshow(u[-1])
    #     # pylab.axis("off")
    #
    #
    # d = [np.mean((im-_u)**2) for _u in u]
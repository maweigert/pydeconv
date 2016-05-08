"""
Krishnan et al

see http://people.ee.duke.edu/~lcarin/NIPS2009_0341.pdf


"""

import numpy as  np
                
from pydeconv._fftw.myfftw import MyFFTW

from pydeconv._hyperlap.lookup_table import ThresholdingLookup

from pydeconv.utils import *

def get_lookup(alpha):
    import os
    import pickle

    cache_dir = os.path.expanduser("~/.pydeconv/cache/hyperlap")
    fname = os.path.join(cache_dir,"lookup_%s.pkl"%alpha)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if not os.path.exists(fname):
        look = ThresholdingLookup(alpha=alpha,v_max= 2., n_v = 512,
                                  b_max=400.,n_b = 512)
        pickle.dump(look,open(fname,"w"))
        return look
    else:
        return pickle.load(open(fname,"r"))



def finite_deriv_dft_forward(dshape,units = None, use_rfft = False):
    """ the dft of the forward finite differences in 2d
    i.e. the fft of [1,-1]
    """

    if units is None:
        units = (1.,)*len(dshape)

    kxs = [np.fft.fftfreq(_s,_u) for _s,_u in zip(dshape,units)]

    if use_rfft:
        kxs[-1] = kxs[-1][:dshape[-1]//2+1]

    if len(kxs)>1:
        KXs = np.meshgrid(*kxs,indexing="ij")
    else:
        KXs = kxs

    return [(np.exp(2.j*np.pi*_K)-1.)/np.prod(units) for _K in KXs]

def deconv_hyperlap(y, h,
                    lam = 1000.,
                    outeriter= 6,
                    inneriter = 1,
                    reg0=0.,
                    alpha = None,
                    n_threads = 6,
                    logged = False):
    
    """ hyper laplacian regularized deconvolution

    Krishnan et al
    see http://people.ee.duke.edu/~lcarin/NIPS2009_0341.pdf

    y is/are the recorded image(s)
    h is/are the (already fftshifted) kernel(s) of same shape

    both can be equal sized tuples, in which case a joint deconvolution is performed

    if the data or the kernels are already fourier transformed then
    set y_is_fft/h_is_fft to True
    
    """



    # see the paper for definition of the following hyperparameter
    beta = 1.
    beta_inc = 2.8

    if alpha is not None:
        look = get_lookup(alpha)
    else:
        alpha = 1.

    #make everything scale invariant
    y_mean = np.mean(y)
    y = y/y_mean

    dshape = y.shape
    FFTW = MyFFTW(dshape,unitary = False, n_threads = n_threads)


    H = FFTW.rfftn(h)
    Y = FFTW.rfftn(y)

    fs = finite_deriv_dft_forward(dshape, use_rfft=True)

    x = 1.*y


    den1  = reduce(np.add,[f.conjugate()*f for f in fs])
    den2 = H.conjugate()*H


    nom1 = H.conjugate()*Y


    res = []

    def objective(x):
        dxs = np.gradient(x)
        X = FFTW.rfftn(x)
        x_b = FFTW.irfftn(X*H)
        ener = .5*lam*np.sum((x_b-y)**2)
        grad = np.sum([abs(dx)**alpha for dx in dxs])
        return ener +grad


    for i in xrange(outeriter):
        print "iteration: %d / %d"%(i+1,outeriter)
        for _ in xrange(inneriter):
            # w subproblem
            #dx1, dx2 = np.gradient(x)
            #dxs = [.5*(np.roll(x,-1,i)-np.roll(x,1,i)) for i in range(y.ndim)]

            dxs = np.gradient(x)

            if alpha == 1.:
                ws = [soft_thresh(dx,1./beta) for dx in dxs]
            else:
                ws = [look(dx,beta) for dx in dxs]

            res.append(x)
            # x subproblem
            w_fs = [FFTW.rfftn(w) for w in ws]

            nom2 = reduce(np.add,[f.conjugate()*w_f for f, w_f in zip(fs,w_fs)])


            x = FFTW.irfftn((nom1+1.*lam/beta*nom2)/(reg0*lam/beta+den1+1.*lam/beta*den2))

            # nom = f1.conjugate()*w1_f + f2.conjugate()*w2_f + 1.*lam/beta*H.conjugate()*Y
            # den = reg + f1.conjugate()*f1 + f2.conjugate()*f2 + 1.*lam/beta*H.conjugate()*H
            # x = FFTW.irfftn(nom/den)


            if logged:
                print "objective f = %s"%objective(x)

            # return H.conjugate()*Y

        beta *= beta_inc


    x *= y_mean

    return x, res


def deconv_hyperlap_breg(y, h,
                    lam = 1000.,
                    gamma = 5.,
                    niter = 10,
                    n_threads = 6,
                    logged = False):

    """ hyper laplacian regularized deconvolution via split bregman

    Krishnan et al
    see http://people.ee.duke.edu/~lcarin/NIPS2009_0341.pdf

    y is/are the recorded image(s)
    h is/are the (already fftshifted) kernel(s) of same shape
    both can be equal sized tuples, in which case a joint deconvolution is performed

    if the data or the kernels are already fourier transformed then
    set y_is_fft/h_is_fft to True

    params:
        lam - the higher, the bigger the data term
        gamma - the smaller, the more TV enforced

    """



    alpha = 1.

    #make everything scale invariant
    y_mean = np.mean(y)
    y = y/y_mean

    dshape = y.shape
    FFTW = MyFFTW(dshape,unitary = False, n_threads = n_threads)


    H = FFTW.rfftn(h)
    Y = FFTW.rfftn(y)

    dft_stencil = dft_lap(dshape, use_rfft=True)

    x = 1.*y


    def objective(x):
        dxs = np.gradient(x)
        X = FFTW.rfftn(x)
        x_b = FFTW.irfftn(X*H)
        ener = .5*lam*np.sum((x_b-y)**2)
        grad = np.sum([abs(dx)**alpha for dx in dxs])
        return ener + grad


    d = np.zeros((x.ndim,)+x.shape)
    b = np.zeros((x.ndim,)+x.shape)
    grad_x = np.stack(np.gradient(x))

    for i in xrange(niter):
        print "iteration: %d / %d"%(i+1,niter)


        # d subproblem
        d = soft_thresh(grad_x+b,1./gamma)


        # x subproblem

        nom1 = H.conjugate()*Y
        nom2 = FFTW.rfftn(divergence(d-b))

        den1 = H.conjugate()*H
        den2  = dft_stencil

        x = FFTW.irfftn((1.*lam/gamma*nom1 - nom2)/(1.*lam/gamma*den1 - den2))

        grad_x = np.stack(np.gradient(x))


        # enforcing constraint
        b += grad_x - d

    x *= y_mean
    return x




if __name__ == '__main__':

    import imgtools
    import pylab
    from pydeconv import deconv_rl

    np.random.seed(0)

    sig_level = .1

    x = imgtools.test_images.actin()[:256,:256]

    x = 10.*x/np.amax(x)

    h = np.fft.fftshift(blur_kernel2(x.shape[0],17))

    y = blur(x,h)

    y += .03*np.amax(y)*np.random.normal(0,1.,y.shape)
    y = np.maximum(y,0)

    # h = h**2
    # h *= 1./np.sum(h)


    u = deconv_hyperlap_breg(y,h, 10000.,
                              gamma = 1000.,
                              niter=10, logged = True)

    u0 = deconv_rl(y,h, 5)


    print "ratio of rms (<1 is good): %.2f"%(np.sum((x-u)**2)/np.sum((x-u0)**2))

    res = [x,y,u0,u]


    ax_p = pylab.subplot2grid((2,len(res)),(1,0), colspan=len(res))
    for i,_u in enumerate(res):
        pylab.subplot2grid((2,len(res)),(0,i))
        pylab.imshow(_u,cmap = "hot")
        pylab.axis("off")
        ax_p.plot(_u[:,168],label = str(i))
    ax_p.legend()

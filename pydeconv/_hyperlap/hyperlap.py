""" joint wiener deconvolution """

import numpy as  np
                
from pydeconv._fftw.myfftw import MyFFTW

def soft_thresh(x,lam):
    """
    the solution to
        w = argmin 1/2*(w-x)^2+lam*|w|
    """

    return np.sign(x)*np.maximum(0.,np.abs(x)-lam)



def finite_deriv_dft_central(dshape,units = None, use_rfft = False):
    """ the dft of the central finite differences in 2d

    i.e. the fft of the stencil [1,0,-1]
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

    return [1.j*np.sin(2.*np.pi*_K)/np.prod(units) for _K in KXs]

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
                   n_threads = 6):
    
    """ hyper laplacian regularized deconvolution

    Krishnan et al
    see http://people.ee.duke.edu/~lcarin/NIPS2009_0341.pdf

    y is/are the recorded image(s)
    h is/are the (already fftshifted) kernel(s)

    both can be equal sized tuples, in which case a joint deconvolution is performed

    if the data or the kernels are already fourier transformed then
    set y_is_fft/h_is_fft to True
    
    """

    # see the paper for definition of the following hyperparameter
    beta0 = 1.
    beta_inc = 2.8




    dshape = y.shape
    FFTW = MyFFTW(dshape,unitary = False, n_threads = n_threads)


    H = FFTW.rfftn(h)
    Y = FFTW.rfftn(y)

    fs = finite_deriv_dft_forward(dshape, use_rfft=True)

    x = 1.*y
    beta = beta0

    den1  = reduce(np.add,[f.conjugate()*f for f in fs])
    den2 = H.conjugate()*H


    nom1 = H.conjugate()*Y


    for i in xrange(outeriter):
        print "iteration: %d / %d"%(i+1,outeriter)
        for _ in xrange(inneriter):
            # w subproblem
            #dx1, dx2 = np.gradient(x)
            dxs = [.5*(np.roll(x,-1,i)-np.roll(x,1,i)) for i in range(y.ndim)]


            ws = [soft_thresh(dx,1./beta) for dx in dxs]


            # x subproblem
            w_fs = [FFTW.rfftn(w) for w in ws]



            nom2 = reduce(np.add,[f.conjugate()*w_f for f, w_f in zip(fs,w_fs)])


            x = FFTW.irfftn((nom1+1.*lam/beta*nom2)/(reg0*lam/beta+den1+1.*lam/beta*den2))

            # nom = f1.conjugate()*w1_f + f2.conjugate()*w2_f + 1.*lam/beta*H.conjugate()*Y
            # den = reg + f1.conjugate()*f1 + f2.conjugate()*f2 + 1.*lam/beta*H.conjugate()*H
            # x = FFTW.irfftn(nom/den)



            # return H.conjugate()*Y

        beta *= beta_inc


    return x




if __name__ == '__main__':

    def blur(d,h):
        d_f = np.fft.rfftn(d)
        h_f = np.fft.rfftn(h)
        return np.fft.irfftn(d_f*h_f)


    def blur_kernel2(N,rad):
        k = np.fft.fftfreq(N)
        KY,KX = np.meshgrid(k,k,indexing="ij")
        KR = np.hypot(KX,KY)
        u = 1.*(KR<=1./rad)
        h = np.abs(np.fft.ifftn(u))**2
        h *= 1./np.sum(h)
        return np.fft.fftshift(h)

    def blur_kernel3(N,rad):
        k = np.fft.fftfreq(N)
        KZ,KY,KX = np.meshgrid(k,k,k,indexing="ij")
        KR = np.sqrt(KX**2+KY**2+KZ**2)
        u = 1.*(KR<=1./rad)
        h = np.abs(np.fft.ifftn(u))**2
        h *= 1./np.sum(h)
        return np.fft.fftshift(h)

    def blur_disk(N,rad):
        k = np.arange(N)-N/2.
        KY,KX = np.meshgrid(k,k,indexing="ij")
        KR = np.hypot(KX,KY)
        h = 1.*(KR<=rad)
        h *= 1./np.sum(h)
        return h

    np.random.seed(0)

    sig_level = .1

    # N = 256
    # d0 = np.zeros((N,) * 2, np.float32)
    # ss = (slice(N/5,4*N/5),)*2
    # d0[ss] = 1.

    # x = np.linspace(-1,1,N)
    # Y,X = np.meshgrid(x,x,indexing="ij")
    #
    # d0 *= 1.*(np.sin(2.*np.pi*X/.1*(1.+.6*X))<0.4)*(np.sin(2.*np.pi*Y/.1*(1.+.6*X))<0.4)


    import imgtools


    # d0 = imgtools.test_images.barbara()
    # N = d0.shape[0]
    #
    # h = np.fft.fftshift(blur_kernel2(N,N/15))
    # h = np.fft.fftshift(blur_disk(N,15))

    d0 = imgtools.test_images.droso128().astype(np.float32)
    d0 *= 1./np.amax(d0)

    N = d0.shape[0]

    h = np.fft.fftshift(blur_kernel3(N,N/10))



    #
    # h = np.zeros((N,N))
    # h[N/2-15:N/2+15,N/2-15:N/2+15] = 1.
    # h = np.fft.fftshift(h)/np.sum(h)

    y = blur(d0, h)

    noise = sig_level*np.amax(y)*np.random.uniform(0.,1.,y.shape)
    y += noise

    y *= 1./np.amax(y)
    u = deconv_hyperlap(y,h, reg0 = 0, lam = 1000)


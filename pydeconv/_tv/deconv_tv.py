""" tv deconvolution 2d or 2d version


see http://www.ipol.im/pub/art/2012/g-tvdc/article.pdf

"""
import sys
import numpy as np
from scipy.ndimage.filters import laplace


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



def deconv_tv(datas, psfs, lam, gamma, Niter = 5):
    """ datas/psfs can be a single 2d/3d image or a list """

    if not isinstance(datas, (tuple,list)):
        datas = [datas]
        psfs = [psfs]


    N = len(datas)

    dshape = datas[0].shape
    # drshape = dshape[:-1]+(dshape[-1]//2+1,)
    
    h_fs = [np.fft.rfftn(h) for h in psfs]
    y_fs = [np.fft.rfftn(d) for d in datas]
    
    u = np.mean(datas[0])+0*datas[0]
    
    d = np.zeros((len(dshape),)+dshape)
    b = np.zeros((len(dshape),)+dshape)

    lap  = dft_lap(dshape,use_rfft = True)

    for i in range(Niter):
        print i
        
        # d subproblem
        gu = np.array(np.gradient(u))

        d = soft_thresh(gu+b,1./gamma)

        # u subproblem

        u_f = 1.*lam/gamma*reduce(np.add,[h_f.conjugate()*y_f for h_f,y_f in zip(h_fs, y_fs)])
        u_f -= np.fft.rfftn(divergence(d-b))
        
        u_f /= 1.*lam/gamma * reduce(np.add,[np.abs(h_f)**2 for h_f in h_fs]) - lap

        u = np.maximum(0,np.fft.irfftn(u_f))

        gu = np.array(np.gradient(u))

        # constraint

        b += gu - d

    return u
    
    
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
    

        

if __name__ == '__main__':
    from matplotlib.pyplot import imread
    
    from spimagine import read3dTiff
    im = read3dTiff("data/usaf3d.tif")[200:328,200:328,200:328]
    # im = imread("data/usaf.png")[200:456,100:356]


    im *= 255

    np.random.seed(0)

    # im*= 10
    
    hx = (3.5,1.)
    hs = [psf(im.shape,np.roll(hx,i)) for i in range(im.ndim)]
    
    
    ys = [myconvolve(im ,h) for h in hs]
    
    # y += .1*np.amax(im)*np.random.uniform(0,1.,im.shape)
    for i,y in enumerate(ys):
        ys[i] = np.random.poisson(y.astype(int))
        
    lam = 100.
    gamma = 4.

    u = deconv_tv(ys,hs,lam, gamma, 10)

    # h_f = np.fft.fftn(h)
    # y_f = np.fft.fftn(y)
    
    # u = np.mean(y)+0*y
    
    # d = np.zeros((len(im.shape),)+im.shape)
    # b = np.zeros((len(im.shape),)+im.shape)

    # lap  = dft_lap(im.shape)

    # def wien(y_f,alpha = .1):
    #     u_f = h_f.conjugate()*y_f
    #     u_f /= np.abs(h_f)**2  + alpha
    #     return np.abs(np.fft.ifftn(u_f))

    # def _fun(u , lam):
    #     gu = np.array(np.gradient(u))
    #     bu = myconvolve(u ,h)
    #     l1_part =  np.mean(reduce(np.add,[np.abs(_g) for _g in np.array(np.gradient(u))]))
    #     l2_part = np.mean((bu-y)**2)
    #     return l1_part + .5*lam*l2_part


    # us = []

    # u_w  = wien(y_f,1./lam)
    
    # for i in range(2):
    #     print i, _fun(u,lam)
        
    #     # d subproblem
    #     gu = np.array(np.gradient(u))

    #     d = soft_thresh(gu+b,1./gamma)

    #     # u subproblem

    #     u_f = 1.*lam/gamma*h_f.conjugate()*y_f - np.fft.fftn(divergence(d-b))
    #     u_f /= 1.*lam/gamma * np.abs(h_f)**2 - lap

    #     u = np.abs(np.fft.ifftn(u_f))

    #     gu = np.array(np.gradient(u))

    #     # constraint

    #     b += gu - d

    
    

    #     us.append(u)

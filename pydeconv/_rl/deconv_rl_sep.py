""" separable approx lr """

import numpy as  np
                

import gputools
import imgtools


def deconv_rl_sep(y, h,
              Niter=10,
              Nsep = 1,
              gamma = 1.e-2,
              log_iter = False):

    """ richardson lucy deconvolution

    with separable approximation used

    y is/are the recorded image(s)
    h is/are the kernel(s)

    both can be equal sized tuples, in which case a joint deconvolution is performed

    increase gamma if something explodes....

    """

    # the separable approxes...
    if h.ndim ==2:
        func_sep_series = imgtools.separable_series3
    elif h.ndim ==3:
        func_sep_series = imgtools.separable_series3

    hs = imgtools.separable_series3(h,Nsep)

    #check if we are far off
    h_all = imgtools.separable_approx3(h,Nsep)[-1]
    print "separable approximation relative error: ", np.sqrt(np.mean((h_all-h)**2))/np.amax(h)
    #
    #
    # def _single_lucy_step(d,u_f,h_f,h_flip_f):
    #     tmp = FFTW.irfftn(u_f*h_f)
    #     tmp2 = FFTW.rfftn(d/(tmp+gamma))
    #     tmp3 = FFTW.irfftn(tmp2*h_flip_f)
    #     return tmp3
    #
    #
    #
    # u = np.mean(ys[0])*np.ones_like(ys[0])
    #
    # for i in range(Niter):
    #     if log_iter:
    #         print "deconv_rl step %s/%s"%(i+1,Niter)
    #     U = FFTW.rfftn(u)
    #     us = [_single_lucy_step(y,U,H,H_flip) for y,H,H_flip in zip(ys,Hs,Hs_flip)]
    #
    #     u *= reduce(np.multiply,us)
    #     #return u,us[0]
    #
    # return u


if __name__ == '__main__':

    from matplotlib.pyplot import imread
    from pydeconv.utils import myconvolve, psf

    im = imread("../../tests/data/usaf.png")


    np.random.seed(0)


    hx = (5.,5.)
    h = psf(im.shape,hx)

    h = np.roll(np.roll(h,20,axis=0),10,axis=1)

    g = myconvolve(im ,h)

    g += .01*np.amax(im)*np.random.normal(0,1.,im.shape)


    u = deconv_rl_sep(g,h,Nsep = 2, gamma = 0.1)
"""
benchmark some deconv methods

mweigert@mpi-cbg.de

"""

import numpy as np
import pandas as pd
from pydeconv import deconv_rl, deconv_wiener, deconv_hyperlap
from gputools import convolve_sep2


def metric(orig, estimate):
    return 10.*np.log10(np.mean((orig-np.mean(orig))**2)/np.mean((orig-estimate)**2))
    # return 10.*np.log10(np.amax(orig)**2/np.mean((orig-estimate)**2)


def best_rl(im, y, h):
    ns = [5, 10, 20, 40]
    us = []
    ms = []
    h = np.fft.fftshift(h)
    for n in ns:
        print n
        us.append(deconv_rl(y, h, n))
        ms.append(metric(im, us[-1]))

    ind = np.argmax(ms)
    return us[ind], ms[ind], {"N": ns[ind]}


def best_hyperlap(im, y, h):
    lams = [10, 50, 200, 1000, 5000, 10000]
    us = []
    ms = []
    h = np.fft.fftshift(h)
    for lam in lams:
        print lam
        u = deconv_hyperlap(y, h, lam, outeriter=30)
        u *= np.mean(im)/np.mean(u)

        us.append(u)
        ms.append(metric(im, us[-1]))

    ind = np.argmax(ms)
    return us[ind], ms[ind], {"N": lams[ind]}


def best_wiener(im, y, h):
    gammas = 10.**(-np.arange(2, 10))
    us = []
    ms = []
    h = np.fft.fftshift(h)
    for gamma in gammas:
        print gamma
        u = deconv_wiener(y, h, gamma=gamma)
        u = np.maximum(0,u)
        u *= np.mean(im)/np.mean(u)

        us.append(u)
        ms.append(metric(im, us[-1]))

    ind = np.argmax(ms)
    return us[ind], ms[ind], {"N": gammas[ind]}


if __name__=='__main__':
    from scipy.misc import face, imread
    from gputools import pad_to_shape

    x = np.linspace(-1, 1, 31)
    hx = np.exp(-5.*x**2)
    hy = np.exp(-2.*x**2)
    hx *= 1./np.sum(hx)
    hy *= 1./np.sum(hy)

    im = np.pad(imread("data/usaf.png"), ((32,)*2,)*2, mode="constant").astype(np.float32)
    # im = np.pad(face()[..., 0],((32,)*2,)*2,mode = "constant")

    h = pad_to_shape(np.outer(hy, hx), im.shape)
    y = convolve_sep2(im, hx, hy)
    y = np.random.poisson(y)

    u_rl, m_rl, params_rl = best_rl(im, y, h)
    u_hyp, m_hyp, params_hyp = best_hyperlap(im, y, h)
    u_wien, m_wien, params_wien = best_wiener(im, y, h)

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.clf()
    for i,u in enumerate([im, y, u_wien, u_rl, u_hyp]):
        plt.subplot(1, 5, i+1)
        plt.imshow(u)
        plt.axis("off")

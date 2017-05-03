"""


mweigert@mpi-cbg.de

"""
import numpy as np
from pydeconv import deconv_hyperlap


if __name__ == '__main__':

    from scipy.misc import  imread
    from gputools import pad_to_shape, convolve_sep2

    x = np.linspace(-1, 1, 31)
    hx = np.exp(-5.*x**2)
    hy = np.exp(-2.*x**2)
    hx *= 1./np.sum(hx)
    hy *= 1./np.sum(hy)

    im = np.pad(imread("data/usaf.png"), ((32,)*2,)*2, mode="constant").astype(np.float32)

    h = pad_to_shape(np.outer(hy, hx), im.shape)
    y = convolve_sep2(im, hx, hy)
    y = np.random.poisson(y)


    u, res = deconv_hyperlap(y, np.fft.fftshift(h),1, beta = 1.,  logged = True)


    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.clf()
    plt.imshow(u)
    plt.axis("off")

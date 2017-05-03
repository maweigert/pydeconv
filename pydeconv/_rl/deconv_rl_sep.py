""" separable approx lr """

from __future__ import absolute_import, division, print_function
import numpy as  np
from gputools.separable import separable_series, separable_approx
from gputools import convolve_sep2, convolve_sep3
from six.moves import range


def _conv_sep_approx(data, hs):
    if data.ndim == 2:
        conv_func = convolve_sep2
    elif data.ndim == 3:
        conv_func = convolve_sep3
    else:
        raise ValueError("data should be either 2d or 3d!")

    res = np.zeros(data.shape, np.float32)
    for i, h in enumerate(hs):
        #print("sep blur %s/%s   (%s)" % (i + 1, len(hs), np.prod([np.sum(_h) for _h in h])))
        res += conv_func(data, *h[::-1])

    return res


def deconv_rl_sep(y, h,
                  Niter=10,
                  Nsep=1,
                  gamma=1.e-2,
                  log_iter=False):
    """ richardson lucy deconvolution

    h should NOT be fftshifted!

    with separable approximation used

    y is/are the recorded image(s)
    h is/are the kernel(s)

    both can be equal sized tuples, in which case a joint deconvolution is performed

    increase gamma if something explodes....

    """

    hs = separable_series(h, Nsep)
    if y.ndim == 2:
        hs_flip = separable_series(h[::-1, ::-1], Nsep)
    elif y.ndim == 3:
        hs_flip = separable_series(h[::-1, ::-1, ::-1], Nsep)
    else:
        raise ValueError("data should be either 2d or 3d!")

    # check if we are far off
    h_reco = separable_approx(h, Nsep)[-1]
    print("separable approximation relative error: ", np.sqrt(np.mean((h_reco - h) ** 2)) / np.amax(h))


    #u = np.mean(y) * np.ones_like(y)
    u = y.copy()

    for i in range(Niter):
        if log_iter:
            print("deconv_rl step %s/%s" % (i + 1, Niter))
        tmp = _conv_sep_approx(u, hs)

        tmp = y / (tmp + gamma)

        u *= _conv_sep_approx(tmp, hs_flip)

    return u


if __name__ == '__main__':
    from scipy.misc import ascent
    from pydeconv.utils import myconvolve, psf

    im = ascent().astype(np.float32)
    pad = 32
    im[:pad] = 0
    im[-pad:] = 0
    im[:,:pad] = 0
    im[:,-pad:] = 0

    np.random.seed(0)

    hx = (1., 7.)
    h = np.fft.fftshift(psf((32,32), hx))


    g = _conv_sep_approx(im, separable_series(h,3))

    g = np.maximum(0,g+.00 * np.amax(im) * np.random.normal(0, 1., im.shape))

    u = deconv_rl_sep(g, h, Niter=50, Nsep=2, gamma=0.001)

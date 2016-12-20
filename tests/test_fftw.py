import numpy as np
import numpy.testing as npt
from pydeconv._fftw.myfftw import MyFFTW


def _single_shape(dshape, n_threads = 4):
    np.random.seed(0)

    d_r = np.random.random(dshape).astype(np.float32)
    d_c = (np.random.random(dshape) + 1.j*np.random.random(dshape)).astype(np.complex64)

    rdshape = dshape[:-1]+(dshape[-1]//2+1,)
    d_c2 = (np.random.random(rdshape) + 1.j*np.random.random(rdshape)).astype(np.complex64)

    f =  MyFFTW(dshape, n_threads = n_threads)

    print "testing %s" % str(dshape)

    rtol = 1.e-3
    atol = 1.e-3

    for _ in range(3):
        npt.assert_allclose(f.fftn(d_c), np.fft.fftn(d_c), rtol = rtol, atol = atol)
        npt.assert_allclose(f.ifftn(d_c), np.fft.ifftn(d_c), rtol = rtol, atol = atol)
        npt.assert_allclose(f.rfftn(d_r), np.fft.rfftn(d_r),  rtol = rtol, atol = atol)
        npt.assert_allclose(f.irfftn(d_c2), np.fft.irfftn(d_c2),  rtol = rtol, atol = atol)

if __name__ == '__main__':


    dshape = (128,128)

    for dshape in ((64,128),(128,128),(512,512),(1024,1024),(128,128,128),(128,256,128)):
        _single_shape(dshape, n_threads=20)
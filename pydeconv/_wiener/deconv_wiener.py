""" joint wiener deconvolution """

import numpy as  np
                
from pydeconv._fftw.myfftw import MyFFTW


def deconv_wiener(ys, hs,
                   gamma = 1.e-6,
                   n_threads = 6,
                   y_is_fft = False,
                   h_is_fft = False):
    
    """ wiener deconv

    y is/are the recorded image(s)
    h is/are the (already fftshifted) kernel(s)

    both can be equal sized tuples, in which case a joint deconvolution is performed

    if the data or the kernels are already fourier transformed then
    set y_is_fft/h_is_fft to True
    
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
    
    FFTW = MyFFTW(dshape,unitary = True, n_threads = n_threads)


    if not h_is_fft:
        Hs = [FFTW.rfftn(h) for h in hs]
    else:
        Hs = hs
        
    if not y_is_fft:
        Ys = [FFTW.rfftn(y) for y in ys]
    else:
        Ys = ys
        
    U = reduce(np.add,[_H.conjugate()*_Y for _H,_Y in zip(Hs,Ys)])
    
    U /= 1.*gamma+reduce(np.add,[np.abs(_H)**2 for _H in Hs])

    return FFTW.irfftn(U)


def deconv_wiener2(ys, hs,
                   gamma = 1.e-6,
                   n_threads = 6,
                   y_is_fft = False,
                   h_is_fft = False):

    """ wiener deconv

    y is/are the recorded image(s)
    h is/are the (already fftshifted) kernel(s)

    both can be equal sized tuples, in which case a joint deconvolution is performed

    if the data or the kernels are already fourier transformed then
    set y_is_fft/h_is_fft to True

    noise level is estimated
    gamma is the thikoniv regularizer

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

    FFTW = MyFFTW(dshape,unitary = True, n_threads = n_threads)


    if not h_is_fft:
        Hs = [FFTW.rfftn(h) for h in hs]
    else:
        Hs = hs

    if not y_is_fft:
        Ys = [FFTW.rfftn(y) for y in ys]
    else:
        Ys = ys


    #estimate noise power spectrum
    hs_cut = [np.abs(h_f[0,0])*1.e-6 for h_f in Hs]
    # inds_noise = [2./(1.+np.exp((np.abs(h_f)-h_cut)/h_cut)) for h_cut, h_f in zip(hs_cut,Hs)]
    #
    inds_noise = [1.*(np.abs(h_f)<h_cut) for h_cut, h_f in zip(hs_cut,Hs)]
    inds_signal = [1. - ind_noise for ind_noise in inds_noise]

    #the different power spectra
    ps_y =  [np.abs(y_f)**2 for y_f in Ys]
    ps_signal = [ind*p for ind,p in zip(inds_signal,ps_y)]


    mean_ps_noise  = [np.mean(ind*p) for ind,p in zip(inds_noise,ps_y)]


    ps_noise = [ind_n*p+ind_s*mean_p for ind_n,ind_s,p, mean_p in
        zip(inds_noise,inds_signal,ps_y,mean_ps_noise)]


    filters = [p_signal/(1.*p_signal+p_noise) for p_signal, p_noise in zip(ps_signal,ps_noise)]



    U = reduce(np.add,[_H.conjugate()*_Y*_F for _H,_Y,_F in zip(Hs,Ys,filters)])



    U /= 1.*gamma+reduce(np.add,[np.abs(_H)**2 for _H in Hs])


    return FFTW.irfftn(U)


if __name__ == '__main__':

    pass
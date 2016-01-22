
from pydeconv import deconv_wiener
import imgtools
from scipy.optimize import minimize, fmin

import numpy as np


def blur(d,h):
    d_f = np.fft.rfftn(d)
    h_f = np.fft.rfftn(h)
    return np.fft.irfftn(d_f*h_f)

def rms(x1,x2):
    return np.sqrt(np.mean((x1-x2)**2))


def get_gamma_min(d0,y,h , maxiter=20):
    def err(gamma):
        u =  deconv_wiener(y,h,gamma=gamma)
        u *= np.mean(y)/np.mean(u)

        return rms(d0, u)

    return fmin(err,0.01,maxiter = maxiter)[0]


def blur_kernel(N,rad):
    k = np.fft.fftfreq(N)
    KY,KX = np.meshgrid(k,k,indexing="ij")
    KR = np.hypot(KX,KY)
    u = 1.*(KR<=1./rad)
    h = np.abs(np.fft.ifftn(u))**2
    h *= 1./np.sum(h)
    return np.fft.fftshift(h)


def form_image(sig_level):
    np.random.seed(0)

    N = 256
    d0 = np.zeros((N,) * 2, np.float32)
    ss = (slice(N/5,4*N/5),)*2
    d0[ss] = 1.

    x = np.linspace(-1,1,N)
    Y,X = np.meshgrid(x,x,indexing="ij")

    d0 *= 1.*(np.sin(2.*np.pi*X/.1*(1.+.6*X))<0.4)*(np.sin(2.*np.pi*Y/.1*(1.+.6*X))<0.4)

    h = np.fft.fftshift(blur_kernel(N,N/15))

    y = blur(d0, h)

    noise = sig_level*np.amax(y)*np.random.uniform(0.,1.,y.shape)
    y += noise

    return d0, y, noise, h

def best_wiener_estimate(signal,noise,h):
    s_f = np.fft.rfftn(signal)
    n_f = np.fft.rfftn(noise)
    h_f = np.fft.rfftn(h)
    c_f = np.fft.rfftn(signal+noise)

    filter = np.abs(s_f)**2/(np.abs(s_f)**2+np.abs(n_f)**2)

    gamma = 1.e-20

    u_f = c_f*h_f.conjugate()/(gamma+np.abs(h_f)**2)*filter

    return np.fft.irfftn(u_f)

def calc_wiener_estimate(y,h):
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    y_f = np.fft.rfftn(y)
    h_f = np.fft.rfftn(h)

    #estimate noise power spectrum

    h_cut = np.sum(h)*1.e-8
    indicator_noise = 2./(1.+np.exp(30*(np.abs(h_f)-h_cut)))
    indicator_signal = 1. - indicator_noise


    power_all = np.abs(y_f)**2
    power_signal = indicator_signal*power_all
    nm_noise = np.mean(indicator_noise*power_all)
    power_noise = indicator_noise*power_all+indicator_signal*nm_noise

    filter = power_signal/(1.*power_signal+power_noise)


    gamma = 100.

    u_f = y_f*h_f.conjugate()/(gamma+np.abs(h_f)**2)*filter

    return np.fft.irfftn(u_f)


def gmin_for_sig(sig_level = .1, maxiter=20):

    d0, y, noise, h = form_image(sig_level)
    g = get_gamma_min(d0,y,h,maxiter=maxiter)
    u = deconv_wiener(y,h,g)
    return d0,y,h,u,g


if __name__ == '__main__':

    d0,y,noise, h = form_image(.1)

    u0 = best_wiener_estimate(y-noise,noise,h)
    u1 = calc_wiener_estimate(y,h)
    # d0,y,h,u,g = gmin_for_sig(.1)


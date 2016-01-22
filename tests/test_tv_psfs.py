import numpy as np


from pydeconv import deconv_tv_al, deconv_wiener

from pydeconv.utils import myconvolve, psf

def mse(x,y):
    return np.sqrt(np.mean((x-y)**2))/np.mean(x)
    
def get_best_wiener(y,h,Nws = 50):
    ws = 1.e-6*2**np.arange(Nws)
    return ws[np.argmin([mse(x,deconv_wiener(y,h,w)) for w in ws])]

                     
if __name__ == '__main__':
    np.random.seed(0)
    #2d
    from matplotlib.pyplot import imread

    s = 0.05
    x = imread("data/usaf.png")

    x *= 0
    x[200:400,200:400] = 1.
    
    mu = 10000.
    rho = 10.
    
    scale = 100

    
    x*= 1.*scale
    
    h = psf(x.shape,(6.,6.))

    
    y = myconvolve(x,h)+s*np.amax(x)*np.random.uniform(0,1,x.shape)


    # u = deconv_tv_al([y,y2],[h,h2])

    
    
    u = deconv_tv_al(y,h,mu,rho)

    
    u2 = deconv_wiener(y,h,rho/mu)
    
    

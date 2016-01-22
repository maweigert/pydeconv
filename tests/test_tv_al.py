import numpy as np


from pydeconv import deconv_tv_al, deconv_wiener

from utils import myconvolve, psf


if __name__ == '__main__':
    #2d
    from matplotlib.pyplot import imread

    s = .1
    x = imread("data/usaf.png")

    x*= 255
    
    h = psf(x.shape,(3.,11))


    h = psf(x.shape,(5.,5.))

    h2 = psf(x.shape,(11,3))
    
    y = myconvolve(x,h)+s*np.amax(x)*np.random.uniform(0,1,x.shape)

    y2 = myconvolve(x,h2)+s*np.amax(x)*np.random.uniform(0,1,x.shape)

    # u = deconv_tv_al([y,y2],[h,h2])

    
    u = deconv_tv_al(y,h,10.,1.)

    u2 = deconv_wiener(y,h,0.1)
    
    #3d
    from spimagine import read3dTiff

    x = read3dTiff("data/usaf3d.tif")[100:228,100:228,100:228]
    h = psf(x.shape,(3.,3.,11))

    h2 = psf(x.shape,(11,3,3))
    h3 = psf(x.shape,(3,11,3))
    
    y = myconvolve(x,h)+.1*np.amax(x)*np.random.uniform(0,1,x.shape)
    y2 = myconvolve(x,h2)+.1*np.amax(x)*np.random.uniform(0,1,x.shape)
    y3 = myconvolve(x,h3)+.1*np.amax(x)*np.random.uniform(0,1,x.shape)

    u = deconv_tv_al([y,y2,y3],[h,h2,h3],100.,1.)

    u2 = deconv_wiener([y,y2,y3],[h,h2,h3],0.01)




    

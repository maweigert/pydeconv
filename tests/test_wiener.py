import numpy as np


from pydeconv import deconv_wiener, deconv_wiener2


from utils import myconvolve, psf


if __name__ == '__main__':

    np.random.seed(0)
    
    #2d
    from matplotlib.pyplot import imread

    x = imread("data/usaf.png")
    x = np.pad(x,((256,)*2,)*2,mode= "constant")
    h = psf(x.shape,(3.,11))

    h2 = psf(x.shape,(11,3))
    
    y = myconvolve(x,h)+.3*np.amax(x)*np.random.uniform(0,1,x.shape)

    y2 = myconvolve(x,h2)+.3*np.amax(x)*np.random.uniform(0,1,x.shape)

    u1 = deconv_wiener([y,y2],[h,h2],1.e-6)

    u2 = deconv_wiener2([y,y2],[h2,h],1.e-6)

    # #3d
    # from spimagine import read3dTiff
    #
    # x = read3dTiff("data/usaf3d.tif")[100:228,100:228,100:228]
    # h = psf(x.shape,(3.,3.,11))
    #
    # h2 = psf(x.shape,(11,3,3))
    # h3 = psf(x.shape,(3,11,3))
    #
    # y = myconvolve(x,h)+.1*np.amax(x)*np.random.uniform(0,1,x.shape)
    # y2 = myconvolve(x,h2)+.1*np.amax(x)*np.random.uniform(0,1,x.shape)
    # y3 = myconvolve(x,h3)+.1*np.amax(x)*np.random.uniform(0,1,x.shape)
    #
    # u = deconv_wiener([y,y2,y3],[h,h2,h3],0.01)
    #
    #
    #
    #
    #

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import pdb

def synthguess(a, b, nbins, filename):

    # -- SYNTHESIZED IMAGE
    # read in image, grab pixel dimensions in arcseconds
    pimage, hdr = fits.getdata('DATA/'+filename, header=True)
    pimage = np.squeeze(pimage)
    nx  = hdr['NAXIS1']
    ny  = hdr['NAXIS2']
    xps = np.absolute(hdr['CDELT1']*3600.) #angular width of the pixels in the RA and DEC dimensions
    yps = np.absolute(hdr['CDELT2']*3600.)

    bmaj =  hdr['BMAJ']*3600
    bmin =  hdr['BMIN']*3600

    # construct the radial grid for the image (generic for projected viewing geometry)
    incl = 0.
    PA   = 0.
    xp_,yp_ = np.meshgrid(xps*(np.arange(nx)-(nx/2.-0.5)),yps*(np.arange(ny)-(ny/2.-0.5)))
    ang = np.radians(270.-PA)
    xp  = np.cos(ang)*xp_-np.sin(ang)*yp_
    yp  = np.sin(ang)*xp_+np.cos(ang)*yp_
    rim = np.sqrt(xp*xp+1./np.cos(np.radians(incl))**2*yp*yp) #Radius
    azang = np.arctan2(yp/np.cos(np.radians(incl)), xp) #Polar angle
    azang[azang<0] = azang[azang<0]+2.*np.pi

    # scale surface brightnesses to Jy/arcsec^2 units

    ##    pimage /= xps*yps     #Originally used pixel size
    pimage /= np.pi * bmaj * bmin/ (4. * np.log(2))

    # flatten into a profile
    SBpim = pimage.flatten()
    rpim  = rim.flatten()
    azangim = azang.flatten()

    #Find SNR
    #Identify patches off-beam
    router = np.where((rpim > 3.3) & (rpim < 3.5) & (azangim > np.radians(42.)) & (azangim < np.radians(48.)))
    rinner = np.where((rpim > .5) & (rpim < .7) & (azangim > np.radians(42.)) & (azangim < np.radians(48.)))
#    plt.subplot(2,1,1)
#    plt.plot(rpim[router], SBpim[router],'.')
#    plt.plot(rpim[rinner], SBpim[rinner],'.r')
#    plt.subplot(2,1,2)
#    plt.plot(np.degrees(azangim[router]), SBpim[router],'.')
#    plt.plot(np.degrees(azangim[rinner]), SBpim[rinner],'.r')
#    #plt.show()
#    print 'Outer mean', np.mean(SBpim[router]),' std: ', np.std(SBpim[router])
#    print 'Inner mean', np.mean(SBpim[rinner]),' std: ', np.std(SBpim[rinner])
#    pdb.set_trace()

    # -- DISCRETE DECOMPOSITION from PERFECT IMAGE
    # define the annular bins (as in d3sbFit.py)
    #nbins = 15
    ## b = np.linspace(0.1, 3., num=nbins)
    ## a = np.roll(b, 1)
    ## dpc=140.
    ## rin = 0.01/dpc
    ## a[0] = rin
    cb = 0.5*(a+b)
    ## bins = rin, b

    # calculate the average surface brightness in each bin
    SBdscp = np.zeros_like(cb)
    print 'Binlow Binhigh AvgSB'
    for i in range(nbins):
        SBdscp[i] = np.nanmean(SBpim[((rpim>a[i]) & (rpim<b[i]))])
        print a[i], b[i],  SBdscp[i]

    # plot things
    ## plt.figure(2)
    ## plt.plot(rpim, SBpim, '.r', cb, SBdscp, 'og')
    ## plt.xscale('log')
    ## plt.yscale('log')
    ## plt.show(block='False')

    return SBdscp

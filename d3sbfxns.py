"""
.. module:: d3sbfxns
    :synopsis: Helper functions for D3SB
.. moduleauthor:: Anjali Tripathi
.. moduleauthor:: Sean Andrews
"""
import pdb as pdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def makebins(rin, b):
    """
    Return bin centers for a given choice of b and rin
    """
    a = np.roll(b, 1)
    a[0] = rin
    cb = 0.5*(a+b)
    return cb
    
def sbdata(filename, PA=0., incl=0., offx=0., offy=0.):
    """
    Read in a synthesized image and return 1D SB profile
    :param PA: position angle in degrees
    :param incl: inclination in degrees
    :param off[x|y]: offsets in arcsec >>
    """    

    #Read in synthesized image
    dimage, hdr = fits.getdata(filename, header=True)
    dimage = np.squeeze(dimage)
    RA  = hdr['CDELT1']*(np.arange(hdr['NAXIS1'])-(hdr['CRPIX1']-1))
    DEC = hdr['CDELT2']*(np.arange(hdr['NAXIS2'])-(hdr['CRPIX2']-1))
    omega_beam = np.pi*(3600.**2)*hdr['BMAJ']*hdr['BMIN']/(4.*np.log(2.))

    #Construct the grid for the image
    PAr = np.radians(PA)
    inclr = np.radians(incl)
    RAo, DECo = np.meshgrid(3600.*RA, 3600.*DEC)

    #Deproject image       
    ap = ((RAo-offx)*np.cos(PAr) - (DECo-offy)*np.sin(PAr))/np.cos(inclr)
    dp = (RAo-offx)*np.sin(PAr) + (DECo-offy)*np.cos(PAr)
    imrad = np.sqrt(ap**2 + dp**2)
    dimage *= np.cos(inclr)/omega_beam
      
    #Beam information to return (convert beam attributes to arcsec)
    #Useful for what >>
    beaminfo = omega_beam, hdr['BMAJ']*3600., hdr['BMIN']*3600.

    return imrad, dimage, beaminfo

def sbmeanbin(rin, b, rpim, SBpim):
    """
    Return the mean surface brightness in each bin
    """
    
    nbins = len(b)
    SBdscp = np.zeros_like(b)
    SBstd = np.zeros_like(b)
    b = np.insert(b, 0, rin)    
    for i in range(nbins):
        SBdscp[i] = np.nanmean(SBpim[((rpim>b[i]) & (rpim<b[i+1]))])
        #Check that this needs to be a nanmean, rather than a mean >>
        SBstd[i] = np.std(SBpim[((rpim>b[i]) & (rpim<b[i+1]))])    
    return SBdscp, SBstd

def sbguess(file, rin, b, PA=0., incl=0., offx=0., offy=0., plotting=0, freq=340e9, dsource = 140.):
    """
    Return mean surface brightness in each bin
    """

    #Read in data
    #Add in noise floor estimation capability >>
    rsb, sb, beaminfo = sbdata(file, PA, incl, offx, offy)
    print 'Beam information:'
    omega, bmaj, bmin = beaminfo
    print 'Omega, bmaj, bmin: ', omega, bmaj, bmin
    c = 3e8 #speed of light in m/s
    lam = c/freq
    
    #Get mean and stddev in each bin
    sbbin, sigmabin = sbmeanbin(rin, b, rsb, sb)
    cb = makebins(rin, b)

    #Print recommended cutoff
    #This will not work well for gaps >>
    cutoff = np.abs(sigmabin/sbbin - 1.).argmin()
    print "I would cutoff after ", cb[cutoff]

    #Print results
    for i in range(len(b)):
        print cb[i], sbbin[i], sigmabin[i]

    #Plot results
    if plotting:
        fig1 = plt.figure()
        plt.plot(rsb, sb, '.y')
        plt.loglog(cb, sbbin, 'or')
        plt.errorbar(cb, sbbin, yerr = sigmabin, fmt = 'o')
        
        fig2 = plt.figure()
        plt.plot(cb, np.abs(sigmabin/sbbin), '-.b')
        plt.plot(cb, np.ones_like(cb), ':k')
        plt.show(block='False')
        

             
    return sbbin


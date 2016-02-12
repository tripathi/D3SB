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

#WARNING: PA and incl are in a different order than in MAIN functions

def makebins(rin, b):
    """
    Return bin centers for a given choice of b and rin
    """
    a = np.roll(b, 1)
    a[0] = rin
    cb = 0.5*(a+b)
    return cb
    
def sbdata(filename, PA=0., incl=0., offx=0., offy=0., plotting = 0):
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

    #Plot original image
    if plotting:
        plt.subplot(1,2,1)
        plt.imshow(dimage, cmap='gray', origin='lower', interpolation='nearest',
               vmin=-5e-5, vmax=.6)
        plt.title('Original')
    
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

    #Plot deprojected image
    if plotting:
        plt.subplot(1,2,2)
        plt.imshow(dimage, cmap='gray', origin='lower', interpolation='nearest',
               vmin=-5e-5, vmax=.6)
        plt.title('Deprojected image')
        plt.show(block=False)
    
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
    Return mean surface brightness in each bin.
    A sample usage of the above functions.
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
        plt.savefig('sbcombare.png')
#        plt.show(block='False')
             
    return sbbin

def initwalkers(cb, pinit, alleq=0, res=0.):
    """
    Perturb param. estimates to initialize emcee walkers
    Return walker positions
    :param cb: Bin centers
    :param pinit: Parameter estimates
    Any inclination or other params must be put first, before the bins. !!
    """
    nbins = len(cb)
    ndim = len(pinit)
    nwalkers = 4*ndim
    p0 = np.zeros((nwalkers, ndim))

    if alleq == 0:
        nextra = ndim - nbins
        inner = np.where(cb<res)
        outer = np.where(cb>=res)
    
    for walker in range(nwalkers):
        if alleq:
            p0[walker] = pinit * (1.+np.random.uniform(-0.2, 0.2, ndim))
        else:
            if len(inner) > 0:
                p0[walker][nextra+inner[0][0]] = pinit[inner[0][0]] * (1.+np.random.uniform(0, 2))
                p0[walker][np.add(nextra,outer)] = pinit[outer] * (1.+np.random.uniform(-0.2, 2, np.size(outer)))
#                p0[walker][np.add(nextra,inner)] = pinit[inner] * (1.+np.random.uniform(0, 2, np.size(inner))) #This doesn't use a power law like Sean's approach
                m = np.log(p0[walker][nextra+inner[0][0]]/p0[walker][nextra+outer[0][0]])/np.log((cb[inner[0][0]])/(cb[outer[0][0]]))
                p0[walker][np.add(nextra,inner)] = (p0[walker][0]/cb[inner[0][0]]**m) * (cb[inner]**m)
                p0[walker][nextra:] = np.minimum.accumulate(p0[walker][nextra:])
            else:
                p0[walker][np.add(nextra,outer)] = pinit[outer] * (1.+np.random.uniform(-0.2, 2, np.size(outer)))
            p0[walker][0:nextra] = pinit[0:nextra]+(1.+np.random.uniform(-0.2, 0.2, nextra))            
            #This doesn't enforce monotonicity the way Sean does >>      

    return p0


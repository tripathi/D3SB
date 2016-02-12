import numpy as np
import scipy.special as sc

def discretemodel(theta, uvsamples, rin, b, fitproj = 1):
    """
    Return complex visibilities corresponding to binned surface brightness
    :param theta: Inclination params and bin coefficients (Inclination params must be put first, before the bins)
    Expects rbin = np.concatenate([np.array([rin]), b]), bsize, u, & v to be saved as globals
    If running on binned, deprojected visibilities, GLOBAL rho and no inclination parameters in theta
    """
#    incl, PA, offset, w = theta
    u, v = uvsamples                    # in **lambda** units                              
    rbin = np.concatenate([np.array([rin]), b])
    bsize = b.size
    
    if fitproj:
        #unpack the parameters, if projection parameters included
        incl = theta[0]
        PA   = theta[1]
        offx = theta[2]
        offy = theta[3]
        w = theta[4:]

        # convert angles to radians
        inclr = np.radians(incl)
        PAr = 0.5*np.pi-np.radians(PA)
        offrx = offx / arcsec
        offry = offy / arcsec

        # coordinate change to deal with projection, rotation, and shifts
        uprime = (u*np.cos(PAr) + v*np.sin(PAr)) 
        vprime = (-u*np.sin(PAr) + v*np.cos(PAr)) * np.cos(inclr)
        rho = np.sqrt(uprime**2 + vprime**2)/arcsec

        # phase center shift
        shift = np.exp(-2.*np.pi*1.0j*((u*-offrx) + (v*-offry)))
                
    else:
        w = theta        
        shift = 1 #no phase center shift needed
        rho = dpjrho/arcsec

    # re-orient arrays
    wbin = np.append(np.concatenate([np.array([0.0]), w]), 0.)
    ww = wbin-np.roll(wbin, -1)
    intensity = np.delete(ww, bsize+1)
    
    # compute the visibilities
    jarg = np.outer(2.*np.pi*rbin, rho)
    jinc = sc.j1(jarg)/jarg
    vrealnoshift = np.dot(2.*np.pi*rbin**2*intensity, jinc)
    vreal = vrealnoshift*shift # impart phase center shift
    vimag = np.zeros_like(vreal)
    vis = vreal + 1.0j*vimag          
    return vis



import numpy as np
import scipy.special as sc

def d3sbModel(theta, uvsamples, bins):

#    incl = 0.
#    w = theta
    incl = np.deg2rad(theta[0]) #Projected Inclination
    w = theta[1:] #Ring bin amplitudes

    u, v = uvsamples
    udeproj = u * np.cos(incl) #Deproject
    #vdeproj = v * np.cos(incl) #Deproject
    rho  = 1e3*np.sqrt(udeproj**2+v**2)
    #rho = 1e3*np.sqrt(vdeproj**2+u**2)

    rin, b = bins

    rbin = np.concatenate([np.array([rin]), b])
    wbin = np.append(np.concatenate([np.array([0.0]), w]), 0.)
    ww = wbin-np.roll(wbin, -1)
    wgt = np.delete(ww, b.size+1)

    jarg = np.outer(2.*np.pi*rbin, rho/206264.806427)
    jinc = sc.j1(jarg)/jarg

    vis = np.dot(2.*np.pi*rbin**2*wgt, jinc)

    return vis

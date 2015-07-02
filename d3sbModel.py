import numpy as np
import scipy.special as sc

def d3sbModel(theta, uvsamples, bins):

#    incl = theta[0]
#    logw = theta[1:]
###    w = 10.**theta
    w = theta

    u, v = uvsamples
#    rho  = 1e3*np.sqrt((u*np.cos(incl*np.pi/180.))**2+v**2)
    rho = 1e3*np.sqrt(u**2+v**2)

    rin, b = bins

    rbin = np.concatenate([np.array([rin]), b])
    wbin = np.append(np.concatenate([np.array([0.0]), w]), 0.)
    ww = wbin-np.roll(wbin, -1)
    wgt = np.delete(ww, b.size+1)

    jarg = np.outer(2.*np.pi*rbin, rho/206264.806427)
    jinc = sc.j1(jarg)/jarg

    vis = np.dot(2.*np.pi*rbin**2*wgt, jinc)

    return vis

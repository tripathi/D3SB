import numpy as np
from d3sbModel import d3sbModel
import pdb
import matplotlib.pyplot as plt
import gpprior as gp

def lnprob(theta, data, bins):

#    rcoeff = theta[0]
    #incl = theta[0]
    incl = 0.
    a = theta[0]
    l = theta[1]
    weights = theta[2:]

#    if (rcoeff <=0) or (rcoeffn >0.01):
#        return -np.inf


#    if (incl >90.) or (incl <0):
#        return -np.inf

    if (l<np.amin(np.diff(bins[1:])) or l>np.amax(np.diff(bins[1:]))):
        return -np.inf

    if (weights<-20).any() or (weights>20).any():
        return -np.inf

    if (weights<0).any():
        return -np.inf

#    if not (np.allclose(weights, np.sort(weights)[::-1])):
#        print'Not monotonic'
#        return -np.inf

    u, v, dreal, dimag, dwgt = data
    uvsamples = u, v

    mreal = d3sbModel(theta[2:], uvsamples, bins)
    mimag = np.zeros_like(u)


    chi2 = np.sum(dwgt*(dreal-mreal)**2) + np.sum(dwgt*(dimag-mimag)**2)
##    dw = np.diff(weights)
    incl = 0.
#    penal0y = np.sum(dw[1:]*dw[:-1] <0)
#    rcoef1 = 0.001
#    regularization =float(rcoeff*np.shape(dreal)[0]/np.shape(weights)[0])
#    chi2tot = chi2+regularization*penalty
    lnp = -0.5*chi2
    prior = -0.5*gp.calcprior(weights, bins, a, l)
    posterior = lnp + prior

    return posterior

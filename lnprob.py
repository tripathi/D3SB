import numpy as np
from d3sbModel import d3sbModel
import pdb
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval

def lnprob(theta, data, bins):

#    rcoeff = theta[0]
    incl = theta[0]
    pcoeff = theta[1:]
    rin, b = bins
    a = np.roll(b, 1)
    a[0] = rin
    cb = 0.5*(a+b)
    weights = polyval(cb, pcoeff)
    thetamodel =  np.concatenate([np.array([incl]), weights])

#    if (rcoeff <=0) or (rcoeffn >0.01):
#        return -np.inf

    if (incl >90.) or (incl <0):
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

    mreal = d3sbModel(thetamodel, uvsamples, bins)
    mimag = np.zeros_like(u)


    chi2 = np.sum(dwgt*(dreal-mreal)**2) + np.sum(dwgt*(dimag-mimag)**2)
    dw = np.diff(weights)
    penalty = np.sum(dw[1:]*dw[:-1] <0)
    rcoeff = 0.001
    regularization =float(rcoeff*np.shape(dreal)[0]/np.shape(weights)[0])

#    print 'Chi2, regularized/chi2', chi2, penalty*regularization/chi2
#    print 'rcoeff, Penalty term', rcoeff, regularization
#    print 'total', chi2+penalty*regularization

    chi2tot = chi2+regularization*penalty
#*(penalty+1)/(np.shape(weights)[0]+1)
#/(np.shape(weights)[0] - penalty)
#*regularization
    lnp = -0.5*chi2tot
    return lnp

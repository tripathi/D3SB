import numpy as np
from d3sbModel import d3sbModel
import pdb
import matplotlib.pyplot as plt
import gpprior as gp

def lnprob_alone(theta, data, bins):

    weights = theta

#    if (rcoeff <=0) or (rcoeffn >0.01):
#        return -np.inf


#    if (incl >90.) or (incl <0):
#        return -np.inf

    # if (l<np.amin(np.diff(bins[1:])) or l>np.amax(np.diff(bins[1:]))):
    #     return -np.inf

    if (weights<-20).any() or (weights>20).any():
        return -np.inf

    if (weights<0).any():
        return -np.inf

#    if not (np.allclose(weights, np.sort(weights)[::-1])):
#        print'Not monotonic'
#        return -np.inf

    u, v, dreal, dimag, dwgt = data
    uvsamples = u, v

    mreal = d3sbModel(theta, uvsamples, bins)
    mimag = np.zeros_like(u)


    chi2 = np.sum(dwgt*(dreal-mreal)**2) + np.sum(dwgt*(dimag-mimag)**2)
    dw = np.diff(weights)

    penalty = np.sum(dw[1:]*dw[:-1] <0)
    print 'Penalty', penalty
#    rcoef1 = 0.001
#    regularization =float(rcoeff*np.shape(dreal)[0]/np.shape(weights)[0])
#    chi2tot = chi2+regularization*penalty
    lnp = -0.5*chi2

    rin, b1 = bins
    indices = np.arange(b1.size)
    gpbins = rin, indices

    a = .0001
    #.005
    l = 1
    prior = -0.5*gp.calcprior(weights, gpbins, a, l)
    print 'Chi2 ', chi2
    print 'Data points', len(u)
    print 'Reduced Chi 2', chi2/(len(u)-np.shape(bins[1:])[1])
    print 'Prior ', gp.calcprior(weights, gpbins, a, l)
    print 'Posterior', lnp+prior
    posterior = lnp + prior

    return posterior

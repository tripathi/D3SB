import numpy as np
from d3sbModel import d3sbModel
import pdb
import matplotlib.pyplot as plt
import gpprior as gp

def lnprob_alone(theta, data, bins, gpa, gpl):

    weights = theta
    u, v, dreal, dimag, dwgt = data
    uvsamples = u, v

    #Setup for using GP prior on bin index
    rin, b1 = bins
    indices = np.arange(b1.size)
    gpbins = rin, indices


    if (weights<0).any():
        return -np.inf

    mreal = d3sbModel(theta, uvsamples, bins)
    mimag = np.zeros_like(u)

    #Calculate chi^2
    chi2 = np.sum(dwgt*(dreal-mreal)**2) + np.sum(dwgt*(dimag-mimag)**2)
    Ndata = 2.*np.shape(dreal)[0]

    #Calculate penalty
    dw = np.diff(weights)
    penalty = np.sum(dw[1:]*dw[:-1] <0)
    rcoeff = 1.#0.01
    regularization =float(rcoeff*Ndata/np.shape(weights)[0])


    #Calculate prior
    prior = -0.5*gp.calcprior(weights, gpbins, gpa, gpl)


    #Print desired quantities
    print '\n Reduced Chi2: ', chi2/(Ndata-np.shape(bins[1:])[1]), \
            ' Chi2: ', chi2, ' NData: ', len(u)
    print '\n Penalty: ', penalty*regularization, \
            ' Nturns: ', penalty, ' multiplier: ', regularization
    print '\n Prior ', -2.*prior, '/chi2: ', -2.*prior/chi2
    print '\n Penalty/chi2 ', penalty*regularization/chi2
    print '\n Penalty/Prior: ', penalty*regularization/(prior*-2)

    #Final value
    chi2tot = chi2+regularization*penalty
    lnp = -0.5*chi2
    posterior = lnp + prior

    return posterior

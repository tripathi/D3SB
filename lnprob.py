import numpy as np
from d3sbModel import d3sbModel
import pdb
import matplotlib.pyplot as plt

def lnprob(theta, data, bins):

    if (theta<-20).any() or (theta>20).any():
        return -np.inf

    if (theta<0).any():
        return -np.inf

##     if not (np.allclose(theta, np.sort(theta)[::-1])):
## #        print 'Not monotonic'
##         return -np.inf

    u, v, dreal, dimag, dwgt = data
    uvsamples = u, v

    mreal = d3sbModel(theta, uvsamples, bins)
    mimag = np.zeros_like(u)


    chi2 = np.sum(dwgt*(dreal-mreal)**2) + np.sum(dwgt*(dimag-mimag)**2)
    lnp = -0.5*chi2
    return lnp

import numpy as np
from d3sbModel import d3sbModel
import pdb
import matplotlib.pyplot as plt

def lnprob(theta, data, bins):

    incl = theta[0]
    weights = theta[1:]

    if (incl >90.) or (incl <0):
        return -np.inf  
                          
    if (weights<-20).any() or (weights>20).any():
        return -np.inf

    if (weights<0).any():
        return -np.inf

##     if not (np.allclose(weights, np.sort(weights)[::-1])):
## #        print 'Not monotonic'
##         return -np.inf

    u, v, dreal, dimag, dwgt = data    
    uvsamples = u, v
    
    mreal = d3sbModel(theta, uvsamples, bins)
    mimag = np.zeros_like(u)


    chi2 = np.sum(dwgt*(dreal-mreal)**2) + np.sum(dwgt*(dimag-mimag)**2)
    lnp = -0.5*chi2
    return lnp

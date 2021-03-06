import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

#Calculate covariance matrix, given bin locations
#Compute prior: [Atranspose Cinverse A] for a given set of weights

def kexp2(i1, i2, ibins=None, a=None, l=None):
    ri = ibins[i1]
    rj = ibins[i2]
    return a*a * np.exp(-((ri - rj)**2.)/(2.*l*l))

def calccovar(binsin, ain, lin):
    '''
    Calculate the covariance matrix using a squared exponential kernel
    a: normalization amplitude
    l: correlation length
    '''
    nbins = binsin.shape[0]
    cov = np.fromfunction(kexp2,(nbins,nbins), ibins=binsin, a=ain, l=lin, dtype = np.int)
    return cov

def calcprior(iweights, bins, a, l, meanguess = None):
    '''
    Invert covariance matrix and take the product with the weights
    vectors
    '''

    if meanguess is None:
        meanguess = np.zeros_like(iweights)

    rin, b = bins
    cba = np.roll(b, 1)
    cba[0] = rin
    cbt = 0.5*(cba+b) #Formerly cb

    cb = cbt
    weights = iweights - meanguess #Subtracting out the initial guess

    C = calccovar(cb, a, l)

    #Non-Cholesky method
    Cinv = np.linalg.inv(C)
    prior = np.dot(np.dot(weights,Cinv), weights) #Check that this is correct
    (sign,logdet) = np.linalg.slogdet(C)
    if sign < 0: 
        plt.imshow(C, interpolation='nearest', cmap='Blues', origin = 'upper')
        plt.colorbar()
        plt.savefig('reglogdet'+str(sign)+str(np.around(logdet, decimals = 4))+'.png')
        plt.clf() 
    
    if l < np.amin(np.abs(np.diff(cb))):
        print 'L is too small'
    print '*****The logdet is', sign, ' * ', logdet

    # print 'Hi'
    # # plt.savefig('tmp'+str(np.around(clogdet, decimals=1))+'.png')
    # plt.clf()
    # print 'Bye'

    #Cholesky factorization, to invert it
    chofac, flag = cho_factor(C) #ADD EXCEPTION HANDLING FOR LINALGERROR
    
    #Find log det
    clogdet = np.sum(2 * np.log((np.diag(chofac))))


    # maxneg = 250.
    #if (sign < 0):
    #     if (logdet > maxneg):
    #         logdet = maxneg
    #     else:
    #    print logdet, l
        #print "Warning - The determinant is negative, so something has gone wrong"


#    result = prior + log stuff
#RtC-1R _ ln det C + N ln 2pi
    #result = prior + logdet
    #+ cb.shape[0]*np.log(2.*np.pi)

    #result =
    prod = (np.dot(weights, cho_solve((chofac, flag), weights)))


    return prod + clogdet + b.shape[0]*np.log(2.*np.pi)



def priortest():
    b = np.linspace(0.01, 0.5, num = 6)
    weights = 1-b**2.
    binsa = [b[0], b]
#    weights[1], weights[4] = weights[4], weights[1]
    P = calcprior(weights, binsa, 1., 1.)
    print P
    return

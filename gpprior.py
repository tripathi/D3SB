import numpy as np
import pdb
import matplotlib.pyplot as plt

#Calculate covariance matrix, given bin locations
#Compute prior: [Atranspose Cinverse A] for a given set of weights

def kexp2(i1, i2, bins=None, a=None, l=None):
    ri = bins[i1]
    rj = bins[i2]
    return a*a * np.exp(-(ri - rj)**2./(2*l*l))

def calccovar(binsin, ain, lin):
    '''
    Calculate the covariance matrix using a squared exponential kernel
    a: normalization amplitude
    l: correlation length
    '''
    nbins = binsin.shape[0]
    C = np.fromfunction(kexp2,(nbins,nbins), bins=binsin, a=ain, l=lin, dtype = np.int)
    return C
    
def calcprior(weights, bins, a, l):
    '''
    Invert covariance matrix and take the product with the weights
    vectors
    '''
    print 'In calcprior'
    C = calccovar(bins, a, l)

    #Cholesky factorization, to invert it
    #cholfac = np.linalg.cholesky(C)
    #Cholesky solve to get first product
    #Product on the other side

    Cinv = np.linalg.inv(C)

    prior = np.dot(np.dot(weights,Cinv), weights) #Check that this is correct
    (sign,logdet) = np.linalg.slogdet(C)
    if (sign < 0):
        print "Warning - The determinant is negative, so something has gone wrong"
         
#    result = prior + log stuff
#RtC-1R _ ln det C + N ln 2pi
    result = prior + logdet + bins.shape[0]*np.log(2.*np.pi)

    return result

def priortest():
    b = np.linspace(0.01, 0.5, num = 20)
    weights = 1-b**2.
    weights[1], weights[4] = weights[4], weights[1]
    plt.plot(b, weights)
    plt.show()
    P = calcprior(weights, b, 1, b[2]-b[1])
    print P



    return


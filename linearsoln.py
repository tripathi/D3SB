import numpy as np
import scipy.special as sc
from scipy import stats
import matplotlib.pyplot as plt
import pdb
from deprojectVis import deproject_vis
from scipy.linalg import cho_factor, cho_solve

#Squared exponential kernel
def exp2kernel(i1, i2, ibins=None, a=None, l=None):
    ri = ibins[i1]
    rj = ibins[i2]
    return a*a * np.exp(-((ri - rj)**2.)/(2.*l*l)) #Doesn't have nugget term

#Covariance matrix for intensities
def calccovar(binsin, ain, lin):
    '''
    Calculate the covariance matrix using a squared exponential kernel
    a: normalization amplitude
    l: correlation length
    '''
    nbins = binsin.shape[0]
    cov = np.fromfunction(exp2kernel,(nbins,nbins), ibins=binsin, a=ain, l=lin, dtype = np.int)
    return cov



def main():
    ##1 - Data setup
    ##We have Ndata data visibilities (D), with covariance matrix Sigma

    #1a. Read in visibility data
    visfilename = 'DATA/fullA_nf_discrete.vis.npz' ##Set
    datfile = np.load(visfilename)
    Dorig = datfile['Vis']
    uorig = datfile['u']
    vorig = datfile['v']
    rhoorig = np.sqrt(uorig**2. + vorig**2.)
    Dwgtorig = datfile['Wgt']
    #Sigmainv = np.diag(Dwgtorig) #a little unnecessary

    #1b. Deproject (and optionally bin) the visibilities so that they're face-on
    incl = 50. #deg
    PA = 70. #deg
    offx = -0.3 #arcsec
    offy = -0.2 #arcsec
    #visbins = np.arange(1., np.amax(rhoorig)/1000, 10) #Visibility bins
    nvisbins = 200 ##Set
    if (nvisbins>1):
        #visbins = np.linspace(np.amin(rhoorig)/1000., np.amax(rhoorig)/1000., nvisbins)
        #!!MAY NEED CHANGING!!
        visbins = stats.mstats.mquantiles(rhoorig/1000, np.arange(20)/20.) #Changing the bins to have roughly even numbers of visibilities in each 
        
        rhodeproj, Ddeproj, sigdeproj = deproject_vis([uorig, vorig, Dorig, Dwgtorig], visbins, incl, PA, offx, offy, errtype='scat')
    else:
        rhodeproj, Ddeproj, sigdeproj = deproject_vis([uorig, vorig, Dorig, Dwgtorig], incl=incl, PA=PA, offx=offx, offy=offy)

    #1c. Set data variables (currently only using real parts)
    arcsec = 180./np.pi*3600.
    D = Ddeproj.real
    rho = rhodeproj/arcsec #units of 1/arcsec
    Sigmainv = np.diag(1./np.square(sigdeproj.real))
    Ndata = D.size
    print 'Number of vis is', Ndata, np.shape(rho), np.shape(Sigmainv)

    ##2 - Model Setup
    ##We have a model visibility (M) which uses Nbins annuli

    #Select model annuli radii in arcsec
    rmin = 0.01/140.
    rmax = 1.1
    Nbins = 40
    radii = np.linspace(rmin, rmax, num=Nbins+1) #Currently does NOT use rin
    rleft = radii[:-1]
    rright = radii[1:]
    rcenter = (rleft+rright)/2.

    #M=Xw, where X is only a function of the bins and baselines
    X = np.empty([Ndata,Nbins])
    for j in range(Ndata):
        for i in range(Nbins):
            X[j][i] = 1./rho[j]*(rright[i]*sc.j1(rho[j]*2.*np.pi*rright[i]) - rleft[i]*sc.j1(rho[j]*2.*np.pi*rleft[i]))

    ##3 - Compute linear algebra


    #Calculate uniform prior mean and covariance matrix
    #The mean of the distribution with a uniform prior is wu, with covariance Cu
    Cu = np.dot(np.dot(X.T, Sigmainv), X)
    Cuinv = np.linalg.inv(Cu)
    wu0 = np.dot(Cuinv, np.dot(np.dot(X.T, Sigmainv), D))
    
    #Alternate method without inverse
    wu = np.linalg.solve(Cu, np.dot(np.dot(X.T, Sigmainv), D))
    

    #Calculate the GP covariance matrix (Cw) from the kernel (k), with mean muw
    #The mean of the distribution with this prior is wgp, with variance Cgp
    gpa = .05 #Hyperparameter amplitude
    gpl = .15 #Hyperparameter lengthscale
    #Cw = calccovar(rcenter, gpa, gpl)
    Cwinv = np.linalg.inv(calccovar(rcenter, gpa, gpl))
    #Cgpinv = Cuinv+Cwinv #Covariance matrix
    #del Cgpinv
    Cgp = np.linalg.inv(Cuinv+Cwinv)

    #Calculate the mean to use, for now it's the truth
    flux = 0.12
    sig = 0.6
    incl = 50.
    PA = 70.
    offx = -0.3
    offy = -0.2
    nominal_SB = (sig/rcenter)**0.7 * np.exp(-(rcenter/sig)**2.5)	# fullA distribution
    int_SB = np.trapz(2.*np.pi*nominal_SB*rcenter, rcenter)		# a check on the total flux
    nominal_SB *= flux / int_SB

    muw = nominal_SB #Truth
    wgp = np.dot(Cgp,(np.dot(Cuinv, wu) + np.dot(Cwinv, muw))) #Mean

    fig = plt.figure(1)
    plt.plot(rho, D, '-k', label='Data')
    plt.plot(rho, np.dot(X, nominal_SB), '-m', label= 'Truth')
    plt.plot(rho, np.dot(X, wu), 'ob', label='Uniform prior')
    plt.plot(rho, np.dot(X, wu0), 'sg', label='Uniform prior (orig inv)')
    #plt.plot(rho, np.dot(X, wgp), 'og', label='GP prior')
    plt.legend()
    plt.show(block=False)

    fig = plt.figure(2)
    plt.plot(rcenter, nominal_SB, '-k', label='Truth')
    plt.plot(rcenter, wu, 'ob', label='Uniform')
    plt.plot(rcenter, wu0, 'sg', label='Uniform orig inv')
    #plt.plot(rcenter, wgp, 'og', label='GP')
    plt.legend()
    plt.show(block=False)
    
    
    pdb.set_trace()




    return

if __name__ == "__main__":
    main()

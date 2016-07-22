import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
import pdb
from deprojectVis import deproject_vis

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


#We have Ndata data visibilities (D) that we'd like to solve for.
#The covariance on the data is Sigma
def main():
    visfilename = 'DATA/fullA_nf_discrete.vis.npz' ##Set
    datfile = np.load(visfilename)
    datfile['u'], datfile['v'], datfile['Vis'], datfile['Wgt']

    Dorig = datfile['Vis']
    uorig = datfile['u']
    vorig = datfile['v']
    rhoorig = np.sqrt(uorig**2. + vorig**2.)
    Dwgt = datfile['Wgt']
    #Sigmainv = np.diag(Dwgt) #a little unnecessary



    #Deproject the visibilities so that they're face-on
    #If binning, set these equal to the binned values
    incl = 50.
    PA = 70.
    offx = -0.3
    offy = -0.2

    visbins = np.arange(1., np.amax(rhoorig)/1000, 10)
    #nvisbins = 200 ##Set
    #if (nvisbins>1):
    #    visbins = np.linspace(np.amin(rhoorig)/1000., np.amax(rhoorig)/1000., nvisbins)
    rhodeproj, Ddeproj, sigdeproj = deproject_vis([uorig, vorig, Dorig, Dwgt], visbins, incl, PA, offx, offy, errtype='scat')
    #else:
#        rhodeproj, Ddeproj, sigdeproj = deproject_vis([uorig, vorig, Dorig, Dwgt], incl=incl, PA=PA, offx=offx, offy=offy)
    D = Ddeproj.real
    arcsec = 180./np.pi*3600.
    rho = rhodeproj/arcsec #units of 1/arcsec
    Sigmainv = np.diag(1./np.square(sigdeproj.real))
    #
    Ndata = D.size
    print 'Number of vis is', Ndata, np.shape(rho), np.shape(Sigmainv)

    #We have a model visibility (M) which uses Nbins annuli

    #Select model annuli radii
    rmin = 0.01/140. #units of arcsec
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
            if (j<1):
                print 'r', rright[i], '  |Right', rright[i]*sc.j1(rho[j]*2.*np.pi*rright[i])
                print 'r', rleft[i], '   |Left', rleft[i]*sc.j1(rho[j]*2.*np.pi*rleft[i])
                print 'c', rcenter[i], '  |Diff', (rright[i]*sc.j1(rho[j]*2.*np.pi*rright[i]) - rleft[i]*sc.j1(rho[j]*2.*np.pi*rleft[i]))
            X[j][i] = 1./rho[j]*(rright[i]*sc.j1(rho[j]*2.*np.pi*rright[i]) - rleft[i]*sc.j1(rho[j]*2.*np.pi*rleft[i]))

    #Calculate uniform prior mean and covariance matrix
    #The mean of the distribution with a uniform prior is wu, with covariance Cu

    Cu = np.dot(np.dot(X.T, Sigmainv), X)
    Cuinv = np.linalg.inv(Cu)
    wu = np.dot(Cuinv, np.dot(np.dot(X.T, Sigmainv), D))



    #Calculate the GP covariance matrix (Cw) from the kernel (k), with mean muw
    #The mean of the distribution with this prior is wgp, with variance Cgp

    gpa = .05 #Hyperparameter amplitude
    gpl = .15 #Hyperparameter lengthscale


    Cw = calccovar(rcenter, gpa, gpl)
    Cwinv = np.linalg.inv(Cw)
    Cgpinv = Cuinv+Cwinv #Covariance matrix
    Cgp = np.linalg.inv(Cgpinv)
    del Cgpinv

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

    pdb.set_trace()




    return

if __name__ == "__main__":
    main()

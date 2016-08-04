import numpy as np
import scipy.special as sc
from scipy import stats
import scipy.linalg as scalg
import matplotlib.pyplot as plt
import pdb
from deprojectVis import deproject_vis
from scipy.linalg import cho_factor, cho_solve
import time
from scipy.optimize import minimize
import d3sbfxns as f

#plt.style.use('mystyle')

#Squared exponential kernel
def exp2kernel(i1, i2, ibins=None, a=None, l=None):
    ri = ibins[i1]
    rj = ibins[i2]
#    tol = 1e-6
#    nugget = np.zeros_like(ri)
#    nugget[(ri-rj)<tol] = 1e-16 #AD HOC!        
    return a*a * np.exp(-((ri - rj)**2.)/(2.*l*l))#+nugget

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

#Calculate the evidence as a function of GP amplitude and length
#See Kaisey OneNote from 6/22/16
def calcZ(theta, cb):    
    #Nrings, D, sigma and X are global vars.

    if (Nrings - cb.size) > 1e-6: print 'Size mismatch \n'
    ggpa = theta[0]
    ggpl = theta[1]
    Corig = calccovar(cb, ggpa, ggpl)
    C = Corig + np.amin(Corig)*np.eye(Nrings) #Add nugget term for stability, changes based on params.
    K = Sigma + np.dot(np.dot(X,C),np.transpose(X))
    (sign,logdet) = np.linalg.slogdet(2.*np.pi*K)
    logZ = -.5*(logdet+np.dot(np.dot(np.transpose(D),np.linalg.inv(K)),D))
    if (sign<0): print "Warning, negative determinant"
    return logZ


def main():

    #Flags to adjust
    plotting = True
    plotdebug = False
    plotinv = False

    
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
    nvisbins = 100. ##Set
    if (nvisbins>1):
        #visbins = np.linspace(np.amin(rhoorig)/1000., np.amax(rhoorig)/1000., nvisbins)
        #!!MAY NEED CHANGING!!
        visbins = stats.mstats.mquantiles(rhoorig/1000, np.arange(nvisbins)/nvisbins) #Changing the bins to have roughly even numbers of visibilities in each        
        rhodeproj, Ddeproj, sigdeproj = deproject_vis([uorig, vorig, Dorig, Dwgtorig], visbins, incl, PA, offx, offy, errtype='scat')
    else:
        rhodeproj, Ddeproj, sigdeproj = deproject_vis([uorig, vorig, Dorig, Dwgtorig], incl=incl, PA=PA, offx=offx, offy=offy)

    #1c. Set data variables (currently only using real parts)
    arcsec = 180./np.pi*3600.
    global D, Sigma, X
    D = Ddeproj.real
    rho = rhodeproj/arcsec #units of 1/arcsec
    Sigma = np.diag(np.square(sigdeproj.real))
    Sigmainv = np.diag(1./np.square(sigdeproj.real))
    Ndata = D.size
    print 'Number of vis is', Ndata, np.shape(rho), np.shape(Sigmainv)

    ##2 - Model Setup
    ##We have a model visibility (M) which uses Nrings annuli

    #Select model annuli radii in arcsec
    rmin = 0.01/140.
    rmax = 1.1
    global Nrings
    Nrings = 30
    radii = np.linspace(rmin, rmax, num=Nrings+1) #Currently does NOT use rin
    rleft = radii[:-1]
    rright = radii[1:]
    rcenter = (rleft+rright)/2.

    #M=Xw, where X is only a function of the bins and baselines
    X = np.empty([Ndata,Nrings])
    for j in range(Ndata):
        for i in range(Nrings):
            X[j][i] = 1./rho[j]*(rright[i]*sc.j1(rho[j]*2.*np.pi*rright[i]) - rleft[i]*sc.j1(rho[j]*2.*np.pi*rleft[i]))

            
    ##3 - Compute linear algebra

    #3a. Calculate uniform prior mean (wu) and covariance matrix (Cu)

    Cuorig = np.dot(np.dot(X.T, Sigmainv), X)

    #Add epsu to the diagonal of the matrix to make it more numerically stable
    print 'Cu condition number ', np.linalg.cond(Cuorig)
    epsu = 10**(int(np.log10(np.amin(np.diag(Cuorig))))-2) ##Can Set
    Cu = Cuorig + epsu*np.eye(Nrings)
    print 'Min (Cu diag)', np.amin(np.diag(Cuorig))
    print 'Eps', epsu
    print 'Cu new condition number ', np.linalg.cond(Cu)
    
    #Calculate wu in one of two ways
    # Method 1: With the inverse    
    Cuinv = np.linalg.inv(Cu)
    wu0 = np.dot(Cuinv, np.dot(np.dot(X.T, Sigmainv), D))     
    # Method 2: Solve without inverse
    wu = np.linalg.solve(Cu, np.dot(np.dot(X.T, Sigmainv), D))
    print 'Max difference btwn 2 methods for wu', np.amax(np.fabs(wu-wu0))
    
    
    #3b. Calculate the GP covariance matrix (Cw) from the kernel (k), with mean muw
    gpa = .1 #Hyperparameter amplitude
    gpl = .02 #Hyperparameter lengthscale

    #Calculate the mean to use, for now it's the truth
    flux = 0.12
    sig = 0.6
    incl = 50.
    PA = 70.
    offx = -0.3
    offy = -0.2
    nominal_SB = (sig/rcenter)**0.7 * np.exp(-(rcenter/sig)**2.5)	# fullA distribution
    int_SB = np.trapz(2.*np.pi*nominal_SB*rcenter, rcenter)		# a check on the total flux
    nominal_SB *= flux / int_SB #Truth


    #Image mean
    synthimg = 'DATA/fullA.image.fits'
    rsb, sb, beaminfo = f.sbdata(synthimg, PA, incl, offx, offy)
    sbbin, sigmabin = f.sbmeanbin(rleft[0], rright, rsb, sb)

    
    
    muw = np.zeros_like(nominal_SB) #Mean zero

    #Calculate the covariance matrix
    Cworig = calccovar(rcenter, gpa, gpl)
    print 'Cw condition number ', np.linalg.cond(Cworig)
    
    #Add nugget to diagonal to make it more numerically stable
    epsw = np.amin(Cworig)
    Cw = Cworig + epsw*np.eye(Nrings)
    print 'Min(Cw), Min (Cw diag)', np.amin(Cworig), np.amin(np.diag(Cworig))
    print 'Eps', epsw
    print 'Cw new condition number ', np.linalg.cond(Cw)
    Cwinv = np.linalg.inv(Cw)

    #3c. Calculate GP prior mean (wgp) and covariance matrix (Cgp)
    
    #Method 1: With the inverse
    Cgpinv0 = Cuinv+Cwinv #Covariance matrix
    Cgp = np.linalg.inv(Cuinv+Cwinv)
    wgp0 = np.dot(Cgp,(np.dot(Cuinv, wu0) + np.dot(Cwinv, muw)))

    #Method 2: Solve without inverse
    Cgpinv = np.linalg.solve(Cu, np.eye(Nrings) + np.dot(Cu, Cwinv))
    Cuinvwu = np.linalg.solve(Cu, wu)
    wgp = np.linalg.solve(Cgpinv, Cuinvwu + np.dot(Cwinv, muw))

    if (plotdebug):
        fig = plt.figure(0)
        plt.subplot(121)
        plt.imshow(np.dot(Cgp, Cgpinv0))
        plt.title('Testing inv method: Cgp*Cgpinv')
        plt.colorbar()
    
        plt.subplot(122)
        plt.plot(rcenter, wu, '-k')
        plt.plot(rcenter, np.dot(Cu, Cuinvwu), 'ob')
        plt.title('Testing solve method: wu')
        
    #4 Evaluate output

    #4a Plot output directly
    #Plot visibilities
    if (plotting):
        fig2 = plt.figure(1)
        plt.plot(rho, D, '-k', label='Data')
        plt.plot(rho, np.dot(X, nominal_SB), '-m', label= 'Truth')
        if (plotinv):
            plt.plot(rho, np.dot(X, wu0), 'sc', label='Uniform prior (inv)', alpha = 0.5)
        plt.plot(rho, np.dot(X, wu), 'ob', label='Uniform prior (solve)')
        if (plotinv):
            plt.plot(rho, np.dot(X, wgp0), 'sm', label='GP prior (inv)', alpha = 0.5)
        plt.plot(rho, np.dot(X, wgp), 'or', label='GP prior (solve)')
        plt.ylabel('Visibility [Jy]')
        plt.xlabel('Rho [1/arcsec]') 
        plt.legend()

        #Plot SB
        fig3 = plt.figure(2)
        plt.plot(rcenter, nominal_SB, '-k', label='Truth')
        plt.plot(rcenter, muw, 'xk', label='Mean')
        if (plotinv):
            plt.plot(rcenter, wu0, 'sc', label='Uniform (inv)', alpha = 0.5)
        plt.plot(rcenter, wu, 'ob', label='Uniform (solve)')
        if (plotinv):
            plt.plot(rcenter, wgp0, 'sm', label='GP (inv)', alpha = 0.5)
        plt.plot(rcenter, wgp, 'or', label='GP (solve)')
        plt.ylabel('SB [Jy/arcsec^2]')
        plt.xlabel('Angle [arcsec]') 
        plt.legend()


    #4b Draws from the posteriors
    #tic = time.time()
    ## gpdraws = np.random.multivariate_normal(wgp, Cgp, size=5000) #Choice of Cgp matters (this from method 0)
    ## #toc = time.time()
    ## #print 'GP draws took', round((toc-tic)/60., 3)
    ## post = np.percentile(gpdraws, [16, 50, 84], axis=0)
    ## loerr = post[1]-post[0]
    ## hierr = post[2]-post[1]

    ## if (plotting):        
    ##     fig4 = plt.figure(3)
    ##     plt.title('Draws from wgp')
    ##     plt.ylabel('SB [Jy/arcsec^2]')
    ##     plt.xlabel('Angle [arcsec]')
    ##     plt.plot(rcenter, gpdraws[0], '-y', label='Draws')
    ##     for draw in gpdraws:
    ##         plt.plot(rcenter, draw, '-y', alpha = 0.1, zorder=1)
    ##     plt.plot(rcenter, nominal_SB, '-k', zorder=2, label='Truth')
    ##     plt.plot(rcenter, wgp, 'ob', zorder=3, label='wgp')
    ##     plt.errorbar(rcenter, post[1], yerr = [loerr, hierr], fmt='or', elinewidth=2, zorder=4, alpha = 0.5, label='Posterior quantiles')
    ##     ax = plt.gca()
    ##     ax.set_yscale('log')
    ##     ax.set_xscale('log')
    ##     plt.xlim(.9*rcenter[0], 1.1*rmax)
    ##     plt.legend(loc='best')
    
    #plt.show()


    #5 Optimize hyperparameters
    #    print calcZ([gpa, gpl], rcenter)
    thetaguess = np.array([gpa,gpl])
    opt = minimize(calcZ, thetaguess, args=(rcenter), method='Nelder-Mead', tol=1e-6)
    print opt.x

    #Calculate the covariance matrix
    nCworig = calccovar(rcenter, opt.x[0], opt.x[1])
    print 'Cw condition number ', np.linalg.cond(nCworig)
    
    #Add nugget to diagonal to make it more numerically stable
    nepsw = np.amin(nCworig)
    nCw = nCworig + epsw*np.eye(Nrings)
    print 'Min(Cw), Min (Cw diag)', np.amin(nCworig), np.amin(np.diag(nCworig))
    print 'Eps', nepsw
    print 'Cw new condition number ', np.linalg.cond(nCw)
    nCwinv = np.linalg.inv(nCw)

    #3c. again Calculate GP prior mean (wgp) and covariance matrix (Cgp)

    #Method 2: Solve without inverse
    nCgpinv = np.linalg.solve(Cu, np.eye(Nrings) + np.dot(Cu, nCwinv))
    nCuinvwu = np.linalg.solve(Cu, wu)
    nwgp = np.linalg.solve(nCgpinv, nCuinvwu + np.dot(nCwinv, muw))

    ## plt.plot(rcenter, nominal_SB, '-k', label='Truth')
    ## plt.plot(rcenter, wu, 'ob', label='Uniform (solve)', alpha = 0.6)
    ## plt.plot(rcenter, wgp, 'or', label='GP (solve)')
    ## plt.plot(rcenter, nwgp, 'og', label='GP (hyperparam)')
    ## plt.ylabel('SB [Jy/arcsec^2]')
    ## plt.xlabel('Angle [arcsec]') 
    ## plt.legend()    

    nCgp = np.linalg.inv(Cuinv+nCwinv)
    gpdraws = np.random.multivariate_normal(nwgp, nCgp, size=5000) #Choice of Cgp matters (this from method 0)
    #toc = time.time()
    #print 'GP draws took', round((toc-tic)/60., 3)
    post = np.percentile(gpdraws, [16, 50, 84], axis=0)
    loerr = post[1]-post[0]
    hierr = post[2]-post[1]
       
    fig4 = plt.figure(3)
    plt.title('Draws from wgp')
    plt.ylabel('SB [Jy/arcsec^2]')
    plt.xlabel('Angle [arcsec]')
    plt.plot(rcenter, gpdraws[0], '-y', label='Draws')
    for draw in gpdraws:
            plt.plot(rcenter, draw, '-y', alpha = 0.1, zorder=1)
    plt.plot(rcenter, nominal_SB, '-k', zorder=2, label='Truth')
    plt.plot(rcenter, wgp, 'ob', zorder=3, label='wgp')
    plt.plot(rcenter, nwgp, 'og', zorder=4, label='nwgp')
    plt.errorbar(rcenter, post[1], yerr = [loerr, hierr], fmt='or', elinewidth=2, zorder=4, alpha = 0.5, label='Posterior quantiles')
    plt.plot(rcenter, muw, 'xk', zorder = 5, label='Image mean')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(.9*rcenter[0], 1.1*rmax)
    plt.legend(loc='best')
    
    pdb.set_trace()

    return

if __name__ == "__main__":
    main()

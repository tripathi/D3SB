import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.special as sc
import d3sbfxns as f
import time
from datetime import datetime
import emcee

def readinVis(datafile):
    """
    Extracts visibility data from a CASA and python generated .npz file.
    :param datafile: name of .npz file
    """
    datfile = np.load(datafile)
    return datfile['u'], datfile['v'], datfile['Vis'], datfile['Wgt']#, datfile['freq']  


def initgpa(nwalkers):
    #This is specific to GP amplitude, not generic parameters
    tmp =  np.random.uniform(0,1,nwalkers*2).reshape((nwalkers,2))
    tmp[:,1] = tmp[:,1]*5
    return tmp

def runemcee(pin, nsteps, nthreads, savename, meanw):
    """
    Run emcee
    :param pin: Parameters
    :param nsteps: Number of MCMC steps to take
    :param savename: Output file name prefix
    :param meanw: Mean to use for covariance
    """
    #Number of parameters and walkers, set by p0
    ndim = 2#np.shape(pin)[1]
    nwalkers = 16#ndim * 4
    
    #Initialize GP param walkers
    p0 = initgpa(nwalkers)
    
    #Setup sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, args=[meanw])

    #Write log file
    f = open('analyticemcee.log', 'a')
    FORMAT = '%m-%d-%Y-%H%M'
    f.write(savename+', '+str(ndim)+', '+str(nsteps)+', '+datetime.now().strftime(FORMAT))

    #Run emcee, and time it
    tic = time.time()
    sampler.run_mcmc(p0, nsteps)
    toc = time.time()

    #Display and record run information
    print 'Elapsed emcee run time:', ((toc-tic)/60.)
    print 'Acceptance:', sampler.acceptance_fraction
    f.write(' ,'+str(round((toc-tic)/60., 2))+', '+str(np.round(np.mean(sampler.acceptance_fraction),2))+'\n')
    f.close()

    #Save the results in a binary file
    np.save('mc_'+savename,sampler.chain)

    print 'Done with this emcee run'

    return sampler.chain

#@profile
def lnprob(theta, meanw):
    """
    Compute log posterior for current parameter values
    :param theta: Hyperparameters to be optimized
    :param meanw: Mean to use for covariance 
    """  
    # unpack the parameters
    gpa = theta[0]
    gpl = theta[1]

 
    #PRIORS    
    #Enforce positive amplitude
    if (gpa<0).any():
        return -np.inf

    if (gpl<0).any():
        return -np.inf

    #LIKELIHOOD
    wcalc = calcwtilde(theta, meanw)
    mvis = np.dot(x, wcalc)

    
    #Compute a chi2 value (proportional to log-likelihood)
    #Fix the imaginary component
    chi2 = np.sum( dwgt*(dvis.real-mvis.real)**2) #+ dwgt*(dvis.imag-mvis.imag)**2 )


    # return a log-posterior value
    return -0.5*(chi2)


def analyticmain():

    #################
    # 1. DATA SETUP #
    #################
    #Data input
    hiresvis = 'DATA/fullA.vis.npz'
    global dvis
    global dwgt
    u, v, dvis, dwgt = readinVis(hiresvis)
    synthimg = 'DATA/fullA.image.fits'
    
    #Geometric params [Currently hardcoded]
    incl = 50.
    PA = 70.
    offx = -0.3
    offy = -0.2
    
    #Assign data vec and matrix
    y = dvis.real[:, np.newaxis] #THIS IS ONLY REALS RIGHT NOW!
    sigma = np.diag(1./dwgt.real)

    #Compute rho   
    arcsec = 180./np.pi*3600.
    ## convert angles to radians
    inclr = np.radians(incl)
    PAr = 0.5*np.pi-np.radians(PA)
    offrx = offx / arcsec
    offry = offy / arcsec
    ## coordinate change to deal with projection, rotation, and shifts
    uprime = (u*np.cos(PAr) + v*np.sin(PAr)) 
    vprime = (-u*np.sin(PAr) + v*np.cos(PAr)) * np.cos(inclr)
    rho = np.sqrt(uprime**2 + vprime**2)/arcsec #Do I need this arcsec????
    M = rho.size

    ## phase center shift
    shift = np.exp(-2.*np.pi*1.0j*((u*-offrx) + (v*-offry)))

    #################
    # 2. BINS SETUP #
    #################
    
    #For now, extract bins from a file
    initfile = np.load('init_fullA_22.npz')
    #rleft = np.insert(initfile['b'][:-1],0, initfile['rin'])
    #rright = initfile['b']
    #cb = initfile['cb']
    
    binmin = 0.1
    binmax = 1.1
    rin = .01/140.
    nbins = 20
    b = np.linspace(binmin, binmax, num=nbins-1) 
    rleft = np.insert(b[:-1],0,rin)
    rright = b
    N = rright.size
    global cb
    cb = 0.5*(rleft + rright)

    #################
    # 3. SET MEAN W #
    #################
    
    #Find mean & std.dev. values at bin locations from synthesized image
    #    rsb, sb, beaminfo = f.sbdata(synthimg, PAr, inclr, offrx, offry) #Add plotting argument to see image, if desired
    #sbbin, sigmabin = f.sbmeanbin(rin, b, rsb, sb)


    #Calculate truth before exiting
    flux = 0.12
    sig = 0.6
    incl = 50.
    PA = 70.
    offx = -0.3
    offy = -0.2
    nominal_SB = (sig/cb)**0.7 * np.exp(-(cb/sig)**2.5)	# fullA distribution; where
    # flux=0.12, sig=0.6, i=50, 
    # PA = 70., offs=[-0.3, -0.2]
    int_SB = np.trapz(2.*np.pi*nominal_SB*cb, cb)		# a check on the total flux
    nominal_SB *= flux / int_SB
    
    #####################
    # 4. ANALYTIC SETUP #
    #####################
    
    #Calculate x, V, and w_hat
    global x
    x = np.empty([M,N])
    intermed = np.empty([M,N])
    rhsright = np.empty(M)
    for j in range(M):
        for i in range(N): 
            x[j][i] = 1./rho[j]*(rright[i]*sc.j1(rho[j]*2.*np.pi*rright[i]) - rleft[i]*sc.j1(rho[j]*2.*np.pi*rleft[i]))
            intermed[j][i] = dwgt.real[j]*x[j][i]
            rhsright[j] = dwgt.real[j]*y[j]
    global lhs
    lhs = np.linalg.inv(np.dot(np.transpose(x),intermed))  #This is Vw undertilde inverse
    rhs = np.dot(np.transpose(x), rhsright)
    global result
    result = np.dot(lhs, rhs) #This is w_hat
    
    ################
    # 5. RUN EMCEE #
    ################

    #
    gpaguess = np.array([0.05])#np.random.uniform(0,1,nwalkers)
    nsteps = 10000
    nthreads = 12
    notes = raw_input('Notes? Only give a short phrase to append to filenames\n')
    savename = notes+'_'+str(nbins)

    print 'Going to start emcee'
    chain1 = runemcee(gpaguess, nsteps, nthreads, savename,nominal_SB)

    pdb.set_trace()
    
    #        np.savez('linalgparts', what = result, x=x, y = y, sigmainvdiag = dwgt.real, v = lhs, truth = nominal_SB, bincenter = cb)
        

def calcwtilde(theta, meanw):        
    #Use GP prior

    #Hyperparameters
    gpa = theta[0]
    gam = 1.
    gpl = theta[1]#cb[3]-cb[1]

    #Calculate covariance
    C = calccovar(cb, gpa, gpl, gam)
    Cinv = np.linalg.inv(C)
    Swinv = lhs + Cinv

    #Return new mean (wtilde)
    intermed2 = np.dot(Cinv,meanw)
    return np.dot(np.linalg.inv(Swinv),np.dot(lhs,result)) + np.dot(np.linalg.inv(Swinv),intermed2)

def kexp2(i1, i2, ibins=None, a=None, l=None, gamma=None):
    ri = ibins[i1]
    rj = ibins[i2]
    return ri**gamma*a*a * np.exp(-((ri - rj)**2.)/(2.*l*l)) #Doesn't have cross term

def calccovar(binsin, ain, lin, gamin):
    '''
    Calculate the covariance matrix using a squared exponential kernel
    a: normalization amplitude
    l: correlation length
    '''
    nbins = binsin.shape[0]
    cov = np.fromfunction(kexp2,(nbins,nbins), ibins=binsin, a=ain, l=lin, gamma = gamin, dtype = np.int)
    return cov

if __name__ == "__main__":
    analyticmain()

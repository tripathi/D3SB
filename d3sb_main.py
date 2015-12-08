import pdb
import sys
import time
import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
from getVis import getVis
from getVisALMA import getVisALMA
from synthimage import synthguess
#from lnprob import lnprob
from opt_func import opt_func
import emcee
from emcee.utils import MPIPool
import multiprocessing as mp
import gpprior as gp
from deprojectVis import deprojectVis

#"""
#Usage:
#This file has 2 functions: the main and the emcee driver.
#Inputs are at the top of the main.
#Hardcoded into the emcee driver are: nwalkers, sizecorr, scale.
#"""


def emceeinit(w0, inclin, nbins, nthreads, nsteps, savename, data, dbins, MPI=0, allbinseq=0):
    """Emcee driver function"""

#HARDCODED - Warning. Also bins.
    global incl
    incl = inclin

    #Initialize the MPI-based pool used for parallelization.
    if MPI:
        print MPI
        pool = MPIPool()
        if not pool.is_master():
                # Wait for instructions from the master process.
                pool.wait()
                sys.exit(0)

    #Setup
    ndim = nbins + 1 #Removing inclination as a variable.
    nwalkers = 4*ndim
    p0 = np.zeros((nwalkers, ndim))
    print 'Nbins is now', nbins

    #Needed for fixing unresolved starting balls
    global b1
    global rin
    rin, b1 = dbins


    #Initialize walkers
    radii = np.arange(nbins)
    sizecorr = 1 #Currently Hardcoded; Scaling factor to treat different radii differently
    scale = 0.2 #Currently hardcoded; Fraction of parameter by which it can vary
    for walker in range(nwalkers):
        for rs in radii:
            rand = np.random.uniform(-(w0[rs]*scale*sizecorr), (w0[rs]*scale*sizecorr))
            if (b1[rs] <= res) and (allbinseq >0) :
                rand = np.random.uniform(0, 2.*w0[rs])
            p0[walker][rs+1] = w0[rs] + rand #Make it rs+2, if a & l vary
        # #Initialize a & l
        p0[walker][0] = np.random.uniform(.0001, .5) #When adding back in, make prev statement rs+1
#        while True:
#            p0[walker][1] = np.random.gamma(2., 2.)*np.amax(dbins[1:])/20. + np.amin(np.diff(dbins[1:]))
#            if (p0[walker][1]>=np.amin(dbins[1:]) or p0[walker][1]<=np.amax(dbins[1:])):
#                break

        #THIS IS A PROBLEM FOR THE 1st BIN WITH rin. Also the normalization
#        p0[walker][0] = incl+np.random.uniform(0.85*incl,1.15*incl) #When adding back in, make prev statement rs+1


    #Write emcee perturbation params to log file
    f = open('emceerand.log', 'a')
    FORMAT = '%m-%d-%Y-%H%M'
    f.write(savename+', '+str(nbins)+', '+str(nsteps)+', '+str(scale)+', '+str(sizecorr)+', '+datetime.now().strftime(FORMAT))

    #Model initialization
    u, v, dreal, dimag, dwgt = data
    udeproj = u * np.cos(incl) #Deproject
    rho  =  1e3*np.sqrt(udeproj**2+v**2)
    indices = np.arange(b1.size)
    global gpbins
    gpbins = dbins
#rin, indices
    global rbin
    rbin = np.concatenate([np.array([rin]), b1])
    jarg = np.outer(2.*np.pi*rbin, rho/206264.806427)
    global jinc
    jinc = sc.j1(jarg)/jarg
#    pool = mp.Pool(nthreads-1)

    #Initialize sampler using MPI if necessary
    if MPI:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads)
    print 'Nbins, Ndim', nbins, ndim
    print 'Dbins', dbins

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

    if MPI:
        #Close the processes.
        pool.close()

    print 'Done with this emcee run'

    #Allow user interaction at end, if not using MPI
#    if not MPI:
#        pdb.set_trace()

    return sampler.chain


#################
# MAIN FUNCTION #
#################
def main():

    #Input files
    ALMA = 1 #Is this an ALMA data file
    basename = 'gp_nogap' #Name common to all files in this run
    if ALMA:
        hiresvis = basename + '.combo.noisy.vis.npz'#.340GHz.vis.npz' #Model visibilities
        synthimg = basename + '.combo.noisy.image.fits' #Synthesized image, for guesses
    else:
        hiresvis = basename + '.vis.fits' #Model visibilities
        synthimg = basename + '_1mm.fits' #Synthesized image, for guesses

    #Parameters
    nbins = 30
    dpc = 140. #Distance to source in pc
    binmin = .01 #Where to start bins in arcsec, but will be cutoff at rin
    binmax = 1.3 #Outer bin edge in arcsec
    rin = 0.01/dpc #Inner cutoff in arcsec
    inclguess = 0. #Inclination in degrees

    #Emcee setup parameters
    nsteps = 25000 #Number of steps to take
    nthreads = 12 #Number of threads
    MPIflag = 0 #Use MPI (1) or not (0)

    # Get data
    if ALMA:
        data = getVisALMA('DATA/'+hiresvis)
    else:
        data = getVis('DATA/'+hiresvis) #Used for CARMA visibilities.

    #Get resolution
    u, v, dreal, dimag, dwgt = data
    freq = 340e9 #GHz
    cms=3e8 #m/s
    arcsec = 180./np.pi*3600.
    global res
    res = cms/freq/np.amax(np.sqrt(u**2 + v**2))*arcsec

    # Choose radial bin locations
    btmp = np.linspace(binmin, binmax/3., num=nbins/2) 
    btmp2 = np.logspace(np.log10(binmax/3), np.log10(binmax), num=nbins/2)
    b=np.concatenate([btmp, btmp2[1:]])
    dbins = rin, b
    global bins
    bins = dbins
    a = np.roll(b, 1)
    a[0] = rin
    cb = 0.5*(a+b)
    nbins = np.shape(b)[0]
    print nbins, b
    

    #Find mean values at bin locations from synthesized image
    global wg
    wg = synthguess(a, b, nbins, synthimg)
    w0=wg
    w0[w0<0]=0 #Don't save negative results
    print wg
    pdb.set_trace()
    
    #Truth for this model
    global himage
    rc = 0.7
    himage  =  (cb/rc)**(-0.75) * np.exp(-(cb/rc)**(2.5))
    Ic = 0.054976#For no gap
    himage *=Ic

    #Save initial guesses to file
    filename = 'opt_'+basename+'_linear_'+str(nbins) #this naming scheme could be improved
    np.savez(filename, cb=cb, wg=wg,  w0=w0)

    #Continue from pre-optimized values for last bin choice
#    infile = np.load('opt_'+basename+'_linear_'+str(nbins)+'.npz')

    #Set filename for emcee output.  Non-MPI version asks for user kbd input
    if not MPIflag:
        notes = raw_input('Notes? Only give a short phrase to append to filenames\n')
    else:
        notes=''
    savename = basename+'_'+str(nbins)+'_'+notes

    #Bin visibilities
    newbins = np.arange(1., np.amax(np.sqrt(u**2 + v**2)), 50.)
    binnedvis = deprojectVis('DATA/'+hiresvis, newbins, nu=freq)
    brho, bvisre, bvisim, bsig, bRuv, breal = binnedvis
    binneddata = brho, np.zeros_like(brho), bvisre, bvisim, bsig #Hack to get new u and v


    #Run emcee
    global dreal, dimag, dwgt
    dreal = bvisre
    dimag = bvisim
    dwgt = 1./bsig**2.
    
    initchain = emceeinit(w0, inclguess, nbins, nthreads, 10000, savename+'_mean', binneddata, dbins, MPIflag)
    print "I did my initial inference of 10000 steps on binned visibilities"

    #Flatten chain
    cstart = 0
    ndim = nbins + 1 #CHANGE accordingly
    samplesw0 = initchain[:, cstart:, :].reshape((-1,ndim))
    vcentral = np.percentile(samplesw0, 50, axis=0)
    print vcentral

    ## plt.plot(cb, vcentral[1:],'.')
    ## plt.plot(cb, himage, '-k', alpha=0.4)
    ## plt.plot(cb, wg, 'rs', alpha = 0.2)
    ## ax = plt.gca()
    ## ax.set_xscale('log')
    ## ax.set_yscale('log')
    ## plt.show()
    ## pdb.set_trace()

    #Run again with central values        
    emceeinit(vcentral[1:], inclguess, nbins, nthreads, nsteps, savename+'_mean', data, dbins, MPIflag, allbinseq = 1)

def lnprob(theta):

    a = theta[0]
#    l = theta[1]
    weights = theta[1:]

    l = 2. *res

#    if (l<np.amin(np.diff(b1)) or l>np.amax(np.diff(b1))):
#        return -np.inf

    #if (weights<-20).any() or (weights>20).any():
#        return -np.inf

    if (weights<0).any() or a<0:
        return -np.inf

    mreal = d3sbModel(weights)
    mimag = np.zeros_like(mreal) #Check if this change is ok.

    chi2 = np.sum(dwgt*(dreal-mreal)**2) + np.sum(dwgt*(dimag-mimag)**2)
    lnp = -0.5*chi2
    prior = -0.5*gp.calcprior(weights, gpbins, a, l, himage)
    posterior = lnp + prior

##     dw = np.diff(weights)
##     penalty = np.sum(dw[1:]*dw[:-1] <0)
##     rcoeff = 1.#0.01
##     regularization =float(rcoeff*2.*np.shape(dreal)[0]/np.shape(weights)[0])

##     chi2tot = chi2+regularization*penalty
## #    print 'Extra penalty term/chi2 ', regularization*penalty/chi2
##     posterior = -0.5*chi2tot
#    posterior = lnp + prior

    return posterior

def d3sbModel(theta):


    w = theta #Ring bin amplitudes
    wbin = np.append(np.concatenate([np.array([0.0]), w]), 0.)
    ww = wbin-np.roll(wbin, -1)
    wgt = np.delete(ww, b1.size+1)

    vis = np.dot(2.*np.pi*rbin**2*wgt, jinc)

    return vis


if __name__ == "__main__":
    main()

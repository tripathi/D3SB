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

#"""
#Usage:
#This file has 2 functions: the main and the emcee driver.
#Inputs are at the top of the main.
#Hardcoded into the emcee driver are: nwalkers, sizecorr, scale.
#"""


def emceeinit(w0, inclin, nbins, nthreads, nsteps, savename, data, dbins, MPI=0):
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
            if b1[rs] <= res:
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
    global dreal, dimag, dwgt
    u, v, dreal, dimag, dwgt = data
#    incl = 0.
    udeproj = u * np.cos(incl) #Deproject
    rho  = 1e3*np.sqrt(udeproj**2+v**2)
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
    if not MPI:
        pdb.set_trace()


#################
# MAIN FUNCTION #
#################
def main():

    #Input files
    ALMA = 1 #Is this an ALMA data file
    basename = 'gp_nogap' #Name common to all files in this run
    if ALMA:
        hiresvis = basename + '.combo.noisy.vis.npz'

#.340GHz.vis.npz' #Model visibilities
        synthimg = basename + '.combo.noisy.image.fits' #Synthesized image, for guesses
    else:
        hiresvis = basename + '.vis.fits' #Model visibilities
        synthimg = basename + '_1mm.fits' #Synthesized image, for guesses

    #Parameters
    numbins = 30
    binmin = .01
#0.02 #Where to start bins in arcsec, but will be cutoff at rin

    binmax = 1.3
    ##gpgap?1.6 #Outer bin edge in arcsec
    dpc = 140. #Distance to source in pc
    rin = 0.005
#gpgap0.1/dpc #Inner cutoff in arcsec
#0.01

    inclguess = 0. #Inclination in degrees

    #Emcee setup parameters
    nsteps = 25000 #Number of steps to take
    nthreads = 12 #Number of threads
    MPIflag = 0 #Use MPI (1) or not (0)

#    print 'No GP prior'


    # Get data
    if ALMA:
        data = getVisALMA('DATA/'+hiresvis)
    else:
        data = getVis('DATA/'+hiresvis) #Used for CARMA visibilities.

    #Corrupt/change data as needed
    u, v, dreal, dimag, dwgt = data
    freq = 340e9 #GHz
    cms=3e8 #m/s
    arcsec = 180./np.pi*3600.

    global res
    res = cms/freq/np.amax(np.sqrt(u**2 + v**2))*arcsec

    
    ## mu = 0
    ## sigma = 1./np.sqrt(dwgt)
    ## dwgt = dwgt * 10000.
    ## ##replace existing data with changed data
    ## data = u,v,dreal, dimag, dwgt


    #If using a known model, find model with given basename from DATA/models.log
    #
    ## #Save the true values as rc ftot, gam
    ## with open("DATA/models.log") as openfile:
    ##      for line in openfile:
    ##      	 if basename in line:
    ##             params = line.split(', ')
    ## try:
    ##     rc = float(params[0])
    ## #If the model can't be found, stop execution
    ## except NameError:
    ##     raise NameError,"Unable to find "+basename+" in DATA/models.log"
    ## Ftot = float(params[1])
    ## gam = float(params[2])
    ## print 'Working on file with params: ',rc, Ftot, gam
    ## rout = rc/dpc #Convert to appropriate units

    ##(True) Model surface brightness
    ##rvals = np.logspace(-2.1, .7, num=200) #Radial range chosen here. Denser than the actual input range.
    ##wtrueall = ((2-gam)*Ftot/(2*np.pi*rout**2)) * (rvals/rout)**(-gam)*np.exp(-(rvals/rout)**(2.-gam)) #SB model



    # Choose radial bins and determine initial guesses.
    #The for loop and array are used to run this code snippet for multiple bin sizes.
    binsizes = np.arange(numbins,numbins+1)
    for nbins in binsizes:
        b = np.zeros(nbins)

        #Set bin locations
####        btmp = np.linspace(binmin, 0.35, num = 6)
####        btmp2 = np.linspace(0.35, 0.5, num = 8)
####        btmp3 = np.linspace(0.5, 1.2, num=9)

        ####b = np.concatenate([btmp, btmp2[1:], btmp3[1:]])

        btmp = np.linspace(binmin, binmax/2., num=nbins/2) #gpgap
        btmp2 = np.linspace(binmax/2., binmax, num=nbins/4)#gpgap
        #np.logspace(np.log10(binmin), np.log10(binmax), num=nbins)
        b=np.concatenate([btmp, btmp2[1:]])#gpgap
#        b = np.concatenate([btmp[btmp<0.4], np.array([0.45, 0.6, binmax])])
        #b = np.concatenate([btmp[btmp<0.35], np.array([0.35, 0.45, 0.6, binmax])])
        numbins = np.shape(b) #Changing number of bins
        nbins = numbins[0]
        print numbins, b
        a = np.roll(b, 1)
        a[0] = rin
        cb = 0.5*(a+b)

        dbins = rin, b

        #Calculate the jinc
#        rbin = np.concatenate([np.array([rin]), b])
#        udeproj = u #* np.cos(incl) #Deproject
#        rho  = 1e3*np.sqrt(udeproj**2+v**2)
#        jarg = np.outer(2.*np.pi*rbin, rho/206264.806427)
#        jinc = sc.j1(jarg)/jarg


        #model values at bin locations
        ## wtrue = ((2-gam)*Ftot/(2*np.pi*rout**2)) * (cb/rout)**(-gam)*np.exp(-(cb/rout)**(2.-gam))

        #Find mean values at bin locations from synthesized image
        global wg
        wg = synthguess(a, b, nbins, synthimg)
        w0=wg


        #Truth for this model
        global himage
        rc = 0.7
        himage  =  (cb/rc)**(-0.75) * np.exp(-(cb/rc)**(2.5))
        Ic = 0.054976#For no gap
        himage *=Ic


        #plt.plot(cb, wtrue, 'ok',
        #plt.plot(cb, wg, 'rs')
        #plt.show(block='False')
        #pdb.set_trace()


        #Optimization using Downhill Simplex
        print "Entering minimization"
        ## opt2 = minimize(opt_func, wg, args=(data, dbins), method='Nelder-Mead', options={'maxiter': 100000, 'maxfev': 100000})
        ## w02 = (opt2.x)
        ## print(w02)
        ## print opt2
        ## ## plt.plot(cb, wg, 'rs', markersize=12, alpha=0.4)
        ## ## #    plt.plot(cb, wtrue, 'k', cb, w0, 'bo', w02, 'co')
        ## ## plt.plot(rvals, wtrueall, 'k', cb, w0, 'bo', cb, w02, 'co')
        ## ## plt.xlim(0.013,4.1)
        ## ## #    plt.ylim(1e-9, 5)
        ## ## ax = plt.gca()
        ## ## #    ax.set_yscale('log')
        ## ## #    ax.set_xscale('log')
        ## ## plt.show(block=False)
        print "Left minimization"

        #Save initial guesses to file
        filename = 'opt_'+basename+'_linear_'+str(nbins)

        w0[w0<0]=0 #Don't save negative results
        ##  w02[w02<0]=0

        np.savez(filename, cb=cb, wg=wg,  w0=w0)
        #pdb.set_trace()


    #Continue from pre-optimized values for last bin choice
    #Add for loop back in, if multiple bin runs desired

    infile = np.load('opt_'+basename+'_linear_'+str(nbins)+'.npz')

    #Print initial guesses
    ##print 'Truth ', infile['wtrue']
    print 'Mean ', infile['w0']
    ##print 'Simplex ', infile['w02']


    #Set filename for emcee output.  Non-MPI version asks for user kbd input
    if not MPIflag:
        notes = raw_input('Notes? Only give a short phrase to append to filenames\n')
    else:
        notes=''
    savename = basename+'_'+str(nbins)+'_'+notes

    global bins
    bins = dbins

    #Run emcee
    emceeinit(infile['w0'], inclguess, nbins, nthreads, nsteps, savename+'_mean', data, dbins, MPIflag)


def lnprob(theta):

    a = theta[0]
#    l = theta[1]
    weights = theta[1:]#    a = .005
#.005

    l = 2. *res#    weights = theta


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

import pdb
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
from getVis import getVis
from getVisALMA import getVisALMA
from synthimage import synthguess
from lnprob import lnprob
from opt_func import opt_func
import emcee
from emcee.utils import MPIPool
from numpy.polynomial import polynomial as P

#"""
#Usage:
#This file has 2 functions: the main and the emcee driver.
#Inputs are at the top of the main.
#Hardcoded into the emcee driver are: nwalkers, sizecorr, scale.
#"""

def emceeinit(w0, incl, nbins, nthreads, nsteps, savename, data, dbins, MPI=0, ncoeff = 0):
    """Emcee driver function"""

    #Initialize the MPI-based pool used for parallelization.
    if MPI:
        print MPI
        pool = MPIPool()
        if not pool.is_master():
                # Wait for instructions from the master process.
                pool.wait()
                sys.exit(0)

    #Setup
#    ndim = nbins + 1
    ndim = ncoeff + 1
    nwalkers = 4*ndim
    p0 = np.zeros((nwalkers, ndim))
#    print 'Nbins is now', nbins

    #Initialize walkers
    radii = np.arange(nbins)
    sizecorr = 1 #Currently Hardcoded; Scaling factor to treat different radii differently
    scale = 0.3 #Currently hardcoded; Fraction of parameter by which it can vary
    for walker in range(nwalkers):
        #for rs in radii:
        for rs in np.arange(ncoeff):
            rand = np.random.uniform(-(w0[rs]*scale*sizecorr), (w0[rs]*scale*sizecorr))
        #     if rs < 3:
        #         rand = np.random.uniform(0, 2.*w0[rs])
            p0[walker][rs+1] = w0[rs] + rand
#        p0[walker][0] = np.random.uniform(0, .01) #When adding back in, make prev statement rs+1
        p0[walker][0] = incl+np.random.uniform(0.85*incl,1.15*incl) #When adding back in, make prev statement rs+1


    #Write emcee perturbation params to log file
    f = open('emceerand.log', 'a')
    FORMAT = '%m-%d-%Y-%H%M'
    f.write(savename+', '+str(nbins)+', '+str(nsteps)+', '+str(scale)+', '+str(sizecorr)+', '+datetime.now().strftime(FORMAT))


    #Initialize sampler using MPI if necessary
    if MPI:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data, dbins], pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, args=[data, dbins])

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
    if not MPIflag:
        pdb.set_trace()


#################
# MAIN FUNCTION #
#################
if __name__ == "__main__":

    #Input files
    ALMA = 1 #Is this an ALMA data file
    basename = 'blind2_fo'
    #gap_fo' #Name common to all files in this run
    if ALMA:
        hiresvis = basename + '.340GHz.vis.npz' #Model visibilities
        synthimg = basename + '.combo.noisy.image.fits' #Synthesized image, for guesses
    else:
        hiresvis = basename + '.vis.fits' #Model visibilities
        synthimg = basename + '_1mm.fits' #Synthesized image, for guesses

    #Parameters
    numbins = 40
    binmin = 0.01 #Where to start bins in arcsec, but will be cutoff at rin
    binmax = .75 #Outer bin edge in arcsec
    dpc = 140. #Distance to source in pc
    rin = 0.1/dpc #Inner cutoff in arcsec
    inclguess = 0. #Inclination in degrees

    #Emcee setup parameters
    nsteps = 25000 #Number of steps to take
    nthreads = 12 #Number of threads
    MPIflag = 0 #Use MPI (1) or not (0)
    norders = 5 #Number of orders to use for polynomial


    # Get data
    if ALMA:
        data = getVisALMA('DATA/'+hiresvis)
    else:
        data = getVis('DATA/'+hiresvis) #Used for CARMA visibilities.

    #Corrupt/change data as needed
    ## u, v, dreal, dimag, dwgt = data
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
        btmp = np.linspace(binmin, binmax, num=nbins)
        b = np.concatenate([btmp[btmp<0.35], np.array([0.35, 0.45, 0.6, binmax])])
        numbins = np.shape(b) #Changing number of bins
        nbins = numbins[0]
        print numbins, b
        a = np.roll(b, 1)
        a[0] = rin
        cb = 0.5*(a+b)
        dbins = rin, b

        #model values at bin locations
        ## wtrue = ((2-gam)*Ftot/(2*np.pi*rout**2)) * (cb/rout)**(-gam)*np.exp(-(cb/rout)**(2.-gam))

        #Find mean values at bin locations from synthesized image
        wg = synthguess(a, b, nbins, synthimg)
        w0=wg
    #    plt.plot(cb, wtrue, 'ok', cb, wg, 'rs')
    #    plt.show(block='False')
    #    pdb.set_trace()


        #Optimization using Downhill Simplex
        print "Entering minimization"
        #Initial polynomial fit
        coeffs = P.polyfit(cb, wg, norders-1)
        # plt.plot(cb, wg, 'rs', cb, P.polyval(cb, coeffs), 'b')

        # opt2 = minimize(opt_func, wg, args=(data, dbins), method='Nelder-Mead', options={'maxiter': 100000, 'maxfev': 100000})
        # w02 = (opt2.x)
        # print(w02)
        # print opt2
        # ## plt.plot(cb, wg, 'rs', markersize=12, alpha=0.4)
        # ## #    plt.plot(cb, wtrue, 'k', cb, w0, 'bo', w02, 'co')
        # ## plt.plot(rvals, wtrueall, 'k', cb, w0, 'bo', cb, w02, 'co')
        # ## plt.xlim(0.013,4.1)
        # ## #    plt.ylim(1e-9, 5)
        # ax = plt.gca()
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # plt.show(block=False)
        # pdb.set_trace()
        print "Left minimization"

        #Save initial guesses to file
        filename = 'opt_'+basename+'_linearpoly_'+str(nbins)

        w0[w0<0]=0 #Don't save negative results
        ##  w02[w02<0]=0

        np.savez(filename, cb=cb, wg=wg,  w0=w0, coeffs=coeffs)
        #pdb.set_trace()


    #Continue from pre-optimized values for last bin choice
    #Add for loop back in, if multiple bin runs desired

    infile = np.load('opt_'+basename+'_linearpoly_'+str(nbins)+'.npz')

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


    #Run emcee
    emceeinit(coeffs, inclguess, nbins, nthreads, nsteps, savename+'_mean', data, dbins, MPIflag, norders)
    #emceeinit(infile['w0'], inclguess, nbins, nthreads, nsteps, savename+'_mean', data, dbins, MPIflag, norders)
    ## emceeinit(infile['wtrue'], nbins, nthreads, nsteps, savename+'_model')
    ## emceeinit(infile['w02'], nbins, nthreads, nsteps, savename+'_simplex')

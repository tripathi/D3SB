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

#"""
#Usage: 
#This file has 2 functions: the main and the emcee driver.
#Inputs are at the top of the main.
#Hardcoded into the emcee driver are: nwalkers, sizecorr, scale.
#"""

def emceerestart(p0, nbins, nthreads, nsteps, savename, data, dbins, MPI=0):
    """Emcee restart function"""

        
    #Setup
    ndim = nbins + 1
    nwalkers = 4*ndim
    print 'Nbins is now', nbins
    
    #Write emcee perturbation params to log file
    f = open('emceerand.log', 'a')
    FORMAT = '%m-%d-%Y-%H%M'
    f.write(savename+', '+str(nbins)+', '+str(nsteps)+', '+datetime.now().strftime(FORMAT))


    #Initialize sampler using MPI if necessary
    if MPI:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data, dbins], pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, args=[data, dbins])

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
    basename = 'gap_fo' #Name common to all files in this run
    if ALMA:
        hiresvis = basename + '.340GHz.vis.npz' #Model visibilities
        synthimg = basename + '.combo.noisy.image.fits' #Synthesized image, for guesses
    else:    
        hiresvis = basename + '.vis.fits' #Model visibilities
        synthimg = basename + '_1mm.fits' #Synthesized image, for guesses

    #Parameters
    nbinsinit = 40 
    numbins = 22
    binmin = 0.01 #Where to start bins in arcsec, but will be cutoff at rin
    binmax = .75 #Outer bin edge in arcsec
    dpc = 140. #Distance to source in pc
    rin = 0.1/dpc #Inner cutoff in arcsec
    inclguess = 0. #Inclination in degrees

    #Emcee setup parameters
    nsteps = 20000 #Number of steps to take
    nthreads = 12 #Number of threads
    MPIflag = 0 #Use MPI (1) or not (0)



    # Get data
    if ALMA:
        data = getVisALMA('DATA/'+hiresvis)
    else:
        data = getVis('DATA/'+hiresvis) #Used for CARMA visibilities.



    # Choose radial bins and determine initial guesses.
    #The for loop and array are used to run this code snippet for multiple bin sizes.
    binsizes = np.arange(numbins,numbins+1)
    for nbins in binsizes:
        b = np.zeros(nbins)

        #Set bin locations
        btmp = np.linspace(binmin, binmax, num=nbinsinit)
        b = np.concatenate([btmp[btmp<0.35], np.array([0.35, 0.45, 0.6, binmax])])
        numbins = np.shape(b) #Changing number of bins
        nbins = numbins[0]
        print numbins, b
 #        b = np.linspace(binmin, binmax, num=nbins)
        a = np.roll(b, 1)
        a[0] = rin
        cb = 0.5*(a+b)
        dbins = rin, b


        #should really read in bin information from .npy file

    #Set filename for emcee output.  Non-MPI version asks for user kbd input
    if not MPIflag:
        notes = raw_input('Notes? Only give a short phrase to append to filenames\n')
    else:
        notes=''    
    savename = basename+'_'+str(nbins)+'_'+notes+'_c'


    

    chainw0 = np.load('mc_'+basename+'_'+str(nbins)+'_'+notes+'_mean.npy')
    lastp0 = chainw0[:,np.shape(chainw0)[1]-1,:]

    #Run emcee from last step
    emceerestart(lastp0, nbins, nthreads, nsteps, savename+'_mean', data, dbins, MPIflag)


    

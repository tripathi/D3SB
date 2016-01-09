"""
.. module:: main
    :synopsis: Run D3SB
.. moduleauthor:: Anjali Tripathi
.. moduleauthor:: Sean Andrews
"""
import numpy as np
import scipy.special as sc
import pdb as pdb
import matplotlib.pyplot as plt

def runemcee(p0, nsteps, savename, MPI=0):
    """
    Run emcee
    :param p0: Initial walker positions
    :param nsteps: Number of MCMC steps to take
    :param savename: Output file name prefix
    :param MPI: Boolean whether to use MPI (default MPI=0)
    """
    #Number of parameters and walkers, set by p0
    ndim = np.shape(p0)[1]
    nwalkers = np*4

    #Initialize the MPI-based pool used for parallelization.
    if MPI:
        pool = MPIPool()
        if not pool.is_master():
                # Wait for instructions from the master process.
                pool.wait()
                sys.exit(0)

    #Initialize sampler
    if MPI:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads)

    #Write log file
    f = open('emcee.log', 'a')
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

    if MPI:
        #Close the processes.
        pool.close()

    print 'Done with this emcee run'

    #Allow user interaction at end, if not using MPI
#    if not MPI:
#        pdb.set_trace()

    return sampler.chain

def lnprob(theta, dvis, dwgt):
    """
    Compute log posterior for current parameter values
    :param theta: Inclination params and bin coefficients (Inclination params must be put first, before the bins)
    :param dvis: Complex visibilities
    :param dwgt: Visibility weights, currently only real
    """  
    # unpack the parameters
    incl = theta[0]
    PA   = theta[1]
    offx = theta[2]
    offy = theta[3]
    w = theta[4:]

    #PRIORS    
    #Enforce positive surface brightnesses
    if (w<0).any():
        return -np.inf
    #Calculate penalty, assuming regulariz. param coefficient of 1
    dw = np.diff(w)
    prior = np.sum(dw[1:]*dw[:-1] <0, dtype='float') * 2.*len(dreal) / len(w)

    #LIKELIHOOD
    #Compute a chi2 value (proportional to log-likelihood)
    mvis = discretemodel(theta)
    ## mvis = discretemodel([incl, PA, np.array([offx, offy]), p]) #What Sean does, but seems unnec. complicated
    chi2 = np.sum(((dvis.real-mvis.real)/dwgt)**2 + 
                  ((dvis.imag-mvis.imag)/dwgt)**2) #Currently using dwgt both times >>

    # return a log-posterior value
    return -0.5*(chi2 + prior)

def discretemodel(theta):
    """
    Return complex visibilities corresponding to binned surface brightness
    :param theta: Inclination params and bin coefficients (Inclination params must be put first, before the bins)
    Expects data and bins to be saved as global variables elsewhere in this same file
    Expects rbin = np.concatenate([np.array([rin]), b]), bsize, u, & v
    """
#    incl, PA, offset, w = theta
    # unpack the parameters
    incl = theta[0]
    PA   = theta[1]
    offx = theta[2]
    offy = theta[3]
    w = theta[4:]

    # convert angles to radians
    inclr = np.radians(incl)
    PAr = 0.5*np.pi-np.radians(PA)
    offr = offset * np.pi / (180.*3600.)

    # coordinate change to deal with projection, rotation, and shifts
    uprime = (u*np.cos(PAr) + v*np.sin(PAr)) 
    vprime = (-u*np.sin(PAr) + v*np.cos(PAr)) * np.cos(inclr)
    rho = np.sqrt(uprime**2 + vprime**2) * np.pi / (180.*3600.)

    # re-orient arrays
    wbin = np.append(np.concatenate([np.array([0.0]), w]), 0.)
    ww = wbin-np.roll(wbin, -1)
    intensity = np.delete(ww, bsize+1)

    # compute the visibilities
    jarg = np.outer(2.*np.pi*rbin, rho)
    jinc = sc.j1(jarg)/jarg
    vrealnoshift = np.dot(2.*np.pi*rbin**2*intensity, jinc)
    
    # impart a phase center shift
    shift = np.exp(-2.*np.pi*1.0j*((u*-offr[0]) + (v*-offr[1])))
    vreal = vrealnoshift*shift
    vimag = np.zeros_like(vreal)

    vis = vreal + 1.0j*vimag          
    return model_vis

def main():

#Choose bins (b, rin)


    #Globals needed by discretemodel
    global rbin
    rbin = np.concatenate([np.array([rin]), b]) #~~~
    global bsize
    bsize = b.size
    
    global u, v
    u, v, dreal, dimag, dwgt = data 	# U and V must be in **lambda** units >>

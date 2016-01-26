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
import d3sbfxns as f

def readinVis(datafile):
    """
    Extracts visibility data from a CASA and python generated .npz file.
    :param datafile: name of .npz file
    """
    datfile = np.load(datafile)
    data = datfile['u'], datfile['v'], datfile['Vis'], datfile['Wgt']   
    return data


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
    prior = np.sum(dw[1:]*dw[:-1] <0, dtype='float') * 2.*len(dwgt) / len(w)

    #LIKELIHOOD
    #Compute a chi2 value (proportional to log-likelihood)
    mvis = discretemodel(theta)
    ## mvis = discretemodel([incl, PA, np.array([offx, offy]), p]) #What Sean does, but seems unnec. complicated
    ## chi2 = np.sum(((dvis.real-mvis.real)/dwgt)**2 + 
    ##               ((dvis.imag-mvis.imag)/dwgt)**2) #Currently using dwgt both times >>
    chi2 = np.sum( dwgt*(dvis.real-mvis.real)**2 + dwgt*(dvis.imag-mvis.imag)**2 )                  

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
    """
    Run D3SB, using the inputs defined in this function
    """    
    ############
    # 1.Inputs #
    ############
    basename = 'fullA' #Name common to all files in this run
    freq = 340e9 #Hz
    dpc = 140. #Distance to source in pc
     
    #Parameters    
    inclguess = 49.7 #Inclination in degrees
    PAguess = 70.1 #position angle in degrees
    offxguess = -.3  #offsets in arcsec
    offyguess = -.2 #offsets in arcsec

    plotting = 1

    rin = 0.1/dpc
    
    #Emcee setup parameters
    nsteps = 2000 #Number of steps to take
    nthreads = 12 #Number of threads
    MPIflag = 0 #Use MPI (1) or not (0)
    doingpreinference = 1

    ###################
    # 2. Read in data #
    ###################
    hiresvis = 'DATA/' + basename + '.vis.npz' #Model visibilities
    synthimg = 'DATA/' + basename + '.image.fits' #Synthesized image, for guesses     

    #Read in visibilities
    #To avoid having problems and two different functions between pre-inference and real run,
    #dvis is an argument, not a global. >>
    data = readinVis(hiresvis) #Used for CARMA visibilities.
    global u, v
    u, v, dvis, dwgt = data

    #Read in surface brightness 
    rsb, sb, beaminfo = f.sbdata(synthimg, PAguess, inclguess, offxguess, offyguess)

    ##########################################
    # 3. Set bins & find mean SB in each bin #
    ##########################################
    
    #Compute resolution for use in initializing bins   
    cms=3e8 #c in m/s
    arcsec = 180./np.pi*3600.
    global res
    res = 1./np.amax(np.sqrt(u**2 + v**2))*arcsec #is this correct, or do I needcms/freq/ >>

    rin = 0.2*res

    #Bin parameters
    nbins = 15
    binmin = .1 #Where to start bins in arcsec, but will be cutoff at rin
    binmax = 1.1 #Outer bin edge in arcsec
    rin = 0.01/dpc #Inner cutoff in arcsec
    
    # Choose radial bin locations (b, rin)   

    # Use Sean's bin location choice >>>
    b = np.linspace(binmin, binmax, num=nbins)

# - ASSIGN BINS
## nbbins = 24
## b1 = 0.015 + 0.037*np.arange(10)
## b2 = np.logspace(np.log10(b1[-1]), np.log10(1.1), num=15)
## bb = np.concatenate([b1[:-1], b2])
## ba = np.roll(bb, 1)
## ba[0] = 0.1/140.
## br = 0.5*(ba+bb)
## bins = 0.1/140., bb

    
    
    #Find mean & std.dev. values at bin locations from synthesized image
    sbbin, sigmabin = f.sbmeanbin(rin, b, rsb, sb)
    cb = f.makebins(rin, b)

    # Need to deal with bins without data that yield NaN >>    
#    w0[w0<0]=0 #Don't save negative results >>

    #Print results
    print 'Bin, SB Mean, SB Stddev'
    for i in range(len(b)):
        print cb[i], sbbin[i], sigmabin[i]
        
    #Plot results
    if plotting:
        fig1 = plt.figure()
        plt.plot(rsb, sb, '.y')
        plt.loglog(cb, sbbin, 'or')
        plt.errorbar(cb, sbbin, yerr = sigmabin, fmt = 'o')
        plt.xlim(binmin/10, binmax*2.)
#        plt.show()
        plt.savefig(basename+'bins.png')
                
        fig2 = plt.figure()
        plt.plot(cb, np.abs(sigmabin/sbbin), '-.b')
        plt.plot(cb, np.ones_like(cb), ':k')
#        plt.show()
        plt.savefig(basename+'sbcombare.png')
        
        
    
    #Globals needed by discretemodel
    global rbin, bsize
    rbin = np.concatenate([np.array([rin]), b]) #~~~
    bsize = b.size
    





    #Save initial guesses to file
    filename = 'init_'+basename+'_'+str(nbins) #this naming scheme could be improved

    proj = PAguess, inclguess, offxguess, offyguess
    np.savez(filename, cb=cb, sbbin=sbbin, proj = proj, rin=rin, b=b)




    pdb.set_trace()


    


    

   
if __name__ == "__main__":
    main()

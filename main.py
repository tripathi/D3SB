"""
.. module:: main
    :synopsis: Run D3SB
.. moduleauthor:: Anjali Tripathi
.. moduleauthor:: Sean Andrews
"""
import time
from datetime import datetime
import numpy as np
import scipy.special as sc
import pdb as pdb
import matplotlib.pyplot as plt
import d3sbfxns as f
from deprojectVis import deproject_vis
import emcee
from emcee.utils import MPIPool

def readinVis(datafile):
    """
    Extracts visibility data from a CASA and python generated .npz file.
    :param datafile: name of .npz file
    """
    datfile = np.load(datafile)
    data = datfile['u'], datfile['v'], datfile['Vis'], datfile['Wgt']   
    return data


def runemcee(p0, nsteps, nthreads, savename, dv, dw, fitproj=1, MPI=0):
    """
    Run emcee
    :param p0: Initial walker positions
    :param nsteps: Number of MCMC steps to take
    :param savename: Output file name prefix
    :param MPI: Boolean whether to use MPI (default MPI=0)
    """
    #Number of parameters and walkers, set by p0
    ndim = np.shape(p0)[1]
    nwalkers = ndim*4

    #Initialize the MPI-based pool used for parallelization.
    if MPI:
        pool = MPIPool()
        if not pool.is_master():
                # Wait for instructions from the master process.
                pool.wait()
                sys.exit(0)
              
    #Initialize sampler
    if MPI:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[dv, dw, fitproj], pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, args=[dv, dw, fitproj])
       
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
    np.save('mc'+str(fitproj)+'_'++savename,sampler.chain)

    if MPI:
        #Close the processes.
        pool.close()

    print 'Done with this emcee run'

    #Allow user interaction at end, if not using MPI
#    if not MPI:
#        pdb.set_trace()

    return sampler.chain

def lnprob(theta, dvis, dwgt, fitproj=1):
    """
    Compute log posterior for current parameter values
    :param theta: Inclination params and bin coefficients (Inclination params must be put first, before the bins)
    :param dvis: Complex visibilities
    :param dwgt: Visibility weights, currently only real
    """  
    # unpack the parameters, if projection parameters included
    if fitproj:
        incl = theta[0]
        PA   = theta[1]
        offx = theta[2]
        offy = theta[3]
        w = theta[4:]
    else:
        w = theta
 
    #PRIORS    
    #Enforce positive surface brightnesses
    if (w<0).any():
        return -np.inf
    #Calculate penalty, assuming regulariz. param coefficient of 1
    dw = np.diff(w)
    prior = np.sum(dw[1:]*dw[:-1] <0, dtype='float') * 2.*len(dwgt) / len(w)

    #LIKELIHOOD
    #Compute a chi2 value (proportional to log-likelihood)
    mvis = discretemodel(theta, fitproj)

    ## mvis = discretemodel([incl, PA, np.array([offx, offy]), p]) #What Sean does, but seems unnec. complicated
    ## chi2 = np.sum(((dvis.real-mvis.real)/dwgt)**2 + 
    ##               ((dvis.imag-mvis.imag)/dwgt)**2) #Currently using dwgt both times >>

    chi2 = np.sum( dwgt*(dvis.real-mvis.real)**2 + dwgt*(dvis.imag-mvis.imag)**2 )                  

    # return a log-posterior value
    return -0.5*(chi2 + prior)

def discretemodel(theta, fitproj = 1):
    """
    Return complex visibilities corresponding to binned surface brightness
    :param theta: Inclination params and bin coefficients (Inclination params must be put first, before the bins)
    Expects rbin = np.concatenate([np.array([rin]), b]), bsize, u, & v to be saved as globals
    If running on binned, deprojected visibilities, GLOBAL rho and no inclination parameters in theta
    """
#    incl, PA, offset, w = theta


    if fitproj:
        #unpack the parameters, if projection parameters included
        incl = theta[0]
        PA   = theta[1]
        offx = theta[2]
        offy = theta[3]
        w = theta[4:]

        # convert angles to radians
        inclr = np.radians(incl)
        PAr = 0.5*np.pi-np.radians(PA)
        offrx = offx / arcsec
        offry = offy / arcsec

        # coordinate change to deal with projection, rotation, and shifts
        uprime = (u*np.cos(PAr) + v*np.sin(PAr)) 
        vprime = (-u*np.sin(PAr) + v*np.cos(PAr)) * np.cos(inclr)
        rho = np.sqrt(uprime**2 + vprime**2) * np.pi / (180.*3600.)

        # phase center shift
        shift = np.exp(-2.*np.pi*1.0j*((u*-offrx) + (v*-offry)))
                
    else:
        w = theta        
        shift = 1 #no phase center shift needed
        rho = dpjrho

    # re-orient arrays
    wbin = np.append(np.concatenate([np.array([0.0]), w]), 0.)
    ww = wbin-np.roll(wbin, -1)
    intensity = np.delete(ww, bsize+1)
    
    # compute the visibilities
    jarg = np.outer(2.*np.pi*rbin, rho)
    jinc = sc.j1(jarg)/jarg
    vrealnoshift = np.dot(2.*np.pi*rbin**2*intensity, jinc)
    vreal = vrealnoshift*shift # impart phase center shift
    vimag = np.zeros_like(vreal)
    vis = vreal + 1.0j*vimag          
    return vis

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

    plotting = 0
    
    #Emcee setup parameters
    nsteps = 2000 #Number of steps to take
    nthreads = 1 #Number of threads
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
    plotme = 1
    rsb, sb, beaminfo = f.sbdata(synthimg, PAguess, inclguess, offxguess, offyguess)

    ##########################################
    # 3. Set bins & find mean SB in each bin #
    ##########################################
    
    #Compute resolution for use in initializing bins   
    cms=3e8 #c in m/s
    global arcsec
    arcsec = 180./np.pi*3600.
    global res
    res = 1./np.amax(np.sqrt(u**2 + v**2))*arcsec #is this correct, or do I needcms/freq/ >>

    #Bin parameters
    rin = 0.01/dpc #Inner cutoff in arcsec
    binmin = 0.2*res #Where to start bins in arcsec, but will be cutoff at ri
    binmax = 1.1 #Outer bin edge in arcsec
    linstep = 0.5*res
    lincutoff = 0.4
    nlogbins = 12

    
    # Choose radial bin locations (b, rin)   
    blin = np.arange(binmin, lincutoff, linstep)
    blog = np.logspace(np.log10(lincutoff), np.log10(binmax), num = nlogbins)
    b = np.concatenate([blin, blog])

    
    #Find mean & std.dev. values at bin locations from synthesized image
    sbbin, sigmabin = f.sbmeanbin(rin, b, rsb, sb)
    cb = f.makebins(rin, b)

    # Need to deal with bins without data that yield NaN >>    
#    w0[w0<0]=0 #Don't save negative results >>

    #Print results
    print 'Bin, SB Mean, SB Stddev'
    for i in range(len(b)):
        print cb[i], sbbin[i], sigmabin[i]
        
    #Plot bin choices (Add RMS) >>
    if plotting:
        fig1 = plt.figure()
        ax1 = plt.subplot(2,1,1)
        ax1.plot(rsb, sb, '.y')
        ax1.loglog(cb, sbbin, 'or')
        ax1.errorbar(cb, sbbin, yerr = sigmabin, fmt = 'o')
        ax1.set_xlim(binmin/10, binmax*2.)

        ax2 = plt.subplot(2,1,2, sharex = ax1)
        ax2.plot(cb, np.abs(sigmabin/sbbin), '-.b')
        ax2.plot(cb, np.ones_like(cb), ':k')
        fig1.subplots_adjust(hspace = 0)
        plt.setp([a.get_xticklabels() for a in fig1.axes[:-1]], visible=False)
        plt.show(block=False)
        fig1.savefig(basename+'sbcombare.png')
    
    #Globals needed by discretemodel
    global rbin, bsize
    rbin = np.concatenate([np.array([rin]), b]) #~~~
    bsize = b.size
    nbins = bsize

    print "Press c to continue \n"
    pdb.set_trace()

    #######################
    # 4. Initialize emcee #
    #######################
    
    #Save initial guesses to file
    filename = 'init_'+basename+'_'+str(nbins) #this naming scheme could be improved

    proj = PAguess, inclguess, offxguess, offyguess
    np.savez(filename, rin=rin, b=b, cb = cb, sbbin=sbbin, proj = proj) #cb is redundant >>

    #Set filename for emcee output.  Non-MPI version asks for user kbd input
    if not MPIflag:
        notes = raw_input('Notes? Only give a short phrase to append to filenames\n')
    else:
        notes=''
    savename = basename+'_'+str(nbins)+'_'+notes

    #Run emcee
    p0 = f.initwalkers(cb, sbbin, alleq=0, res=res)
    newbins = np.arange(1., np.amax(np.sqrt(u**2 + v**2)), 50.)
    dprj_vis = deproject_vis([u, v, dvis, dwgt], newbins, inclguess, PAguess, offxguess, offyguess)
    global dpjrho
    dpjrho, dpjvis, dpjsig = dprj_vis

    print 'Pre runemcee'
    pdb.set_trace()

    chain0 = runemcee(p0, 20000, nthreads, savename, dpjvis, 1./dpjsig.real**2., fitproj = 0, MPI=0)

    #p0 = f.initwalkers(cb, np.insert(sbbin, 0, [inclguess, PAguess, offxguess, offyguess]), alleq=0, res=res)

    
    #Flatten chain
    cstart = 0
    ndim = nbins #+ 1 #CHANGE accordingly
    samplesw0 = chain0[:, cstart:, :].reshape((-1,ndim))
    vcentral = np.percentile(samplesw0, 50, axis=0)
    print vcentral

    if plotting:
        ax1.plot(rsb, sb, '.y')
        ax1.loglog(cb, sbbin, 'or')
        ax1.errorbar(cb, sbbin, yerr = sigmabin, fmt = 'o')
        ax1.plot(cb, vcentral,'.')
        plt.show()
        pdb.set_trace()

           
if __name__ == "__main__":
    main()

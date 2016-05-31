def runemcee(p0, nsteps, nthreads, savename, dv, dw, fitproj=1):
    """
    Run emcee
    :param p0: Initial walker positions
    :param nsteps: Number of MCMC steps to take
    :param savename: Output file name prefix
    """
    #Number of parameters and walkers, set by p0
    ndim = np.shape(p0)[1]
    nwalkers = ndim*4


    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, args=[dv, dw])
       
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
    np.save('mc'+str(fitproj)+'_'+savename,sampler.chain)

    print 'Done with this emcee run'

    return sampler.chain

#@profile
def lnprob(theta, dvis, dwgt):
    """
    Compute log posterior for current parameter values
    :param theta: Inclination params and bin coefficients (Inclination params must be put first, before the bins)
    :param dvis: Complex visibilities
    :param dwgt: Visibility weights, currently only real
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
    wcalc = analyticvals
    mvis = np.dot(xi, wcalc)

    
    #Compute a chi2 value (proportional to log-likelihood)
    #Fix the imaginary component
    chi2 = np.sum( dwgt*(dvis.real-mvis.real)**2 #+ dwgt*(dvis.imag-mvis.imag)**2 )


    # return a log-posterior value
    return -0.5*(chi2)


def analyticmean():
    #Data input
    hiresvis = 'DATA/fullA.vis.npz'
    u, v, dvis, dwgt = readinVis(hiresvis)
    synthimg = 'DATA/fullA.image.fits'
    
    #Geometric params [Currently hardcoded]
    incl = 50.
    PA = 70.
    offx = -0.3
    offy = -0.2

    
    #Assign data vec and matrix
    y = dvis.real[:, np.newaxis]
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
    
    #For now, extract bins from a file
    initfile = np.load('init_fullA_22.npz')
    #rleft = np.insert(initfile['b'][:-1],0, initfile['rin'])
    #rright = initfile['b']
    #cb = initfile['cb']
    
    binmin = 0.1
    binmax = 1.1
    rin = .01/140.

    ncycles = 1
    
    for count in range(ncycles):
        tic = time.time()
        nbins = 20#(count+1)*20
        b = np.linspace(binmin, binmax, num=nbins-1) 

        #Find mean & std.dev. values at bin locations from synthesized image
        rsb, sb, beaminfo = f.sbdata(synthimg, PAr, inclr, offrx, offry) #Add plotting argument to see image, if desired
        sbbin, sigmabin = f.sbmeanbin(rin, b, rsb, sb)
    
            
        rleft = np.insert(b[:-1],0,rin)
        rright = b
        cb = 0.5*(rleft + rright)
    
        N = rright.size
    


    #For ease of quick implementation, I'm coding up the function on paper, not the simplification Sean uses
    ## jarg = np.outer(2.*np.pi*rbin, rho)
    #Failed attempt at doing this with vectors
    #    jargleft = np.outer(rho, 2.*np.pi*rleft)
    #    jargright = np.outer(rho, 2.*np.pi*rright)
    #    jright = sc.j1(jargright)
    #    jleft = sc.j1(jargleft)
    ## jinc = sc.j1(jarg)/jarg
    ## vrealnoshift = np.dot(2.*np.pi*rbin**2*intensity, jinc)
    #    x = (rright*sc.j1(jargright) - rleft*sc.j1(jargleft))

    #sigmainv = np.diag(dwgt.real) #Not used
    
    #Calculate x, V, and w_hat
        x = np.empty([M,N])
        intermed = np.empty([M,N])
        rhsright = np.empty(M)
        for j in range(M):
            for i in range(N): 
                x[j][i] = 1./rho[j]*(rright[i]*sc.j1(rho[j]*2.*np.pi*rright[i]) - rleft[i]*sc.j1(rho[j]*2.*np.pi*rleft[i]))
                intermed[j][i] = dwgt.real[j]*x[j][i]
                rhsright[j] = dwgt.real[j]*y[j]

        lhs = np.linalg.inv(np.dot(np.transpose(x),intermed))  #This is Vw undertilde inverse
        rhs = np.dot(np.transpose(x), rhsright)
        result = np.dot(lhs, rhs) #This is w_hat
        
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
        
        np.savez('linalgparts', what = result, x=x, y = y, sigmainvdiag = dwgt.real, v = lhs, truth = nominal_SB, bincenter = cb)
        
        #############
        #With  prior#
        #############
        
        #Hyperparameter guesses
        gpa = 1.
        gam = count*(-.5)
        gpl = cb[3]-cb[1]
        C = calccovar(cb, gpa, gpl, gam)
        Cinv = np.linalg.inv(C)
        Swinv = lhs + Cinv

        checkprod = np.dot (C, Cinv)
        
        intermed2 = np.dot(Cinv,nominal_SB)
        
        wtilde = np.dot(np.linalg.inv(Swinv),np.dot(lhs,result)) + np.dot(np.linalg.inv(Swinv),intermed2)
        toc = time.time()
        print 'Time taken is ', toc-tic
        print 'Wtilde is:', wtilde    

#def calcwtilde():

#Make x global        

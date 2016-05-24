import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.special as sc
import d3sbfxns as f

def readinVis(datafile):
    """
    Extracts visibility data from a CASA and python generated .npz file.
    :param datafile: name of .npz file
    """
    datfile = np.load(datafile)
    return datfile['u'], datfile['v'], datfile['Vis'], datfile['Wgt']#, datfile['freq']   

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
    

def main():


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
        nbins = (count+1)*10
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
        gpa = 2.
        gam = -.5
        gpl = cb[4]-cb[1]
        C = calccovar(cb, gpa, gpl, gam)
        Cinv = np.linalg.inv(C)
        Swinv = lhs + Cinv
        
        intermed2 = np.dot(Cinv,sbbin)
        
        wtilde = np.dot(np.linalg.inv(Swinv),np.dot(lhs,result)) + np.dot(np.linalg.inv(Swinv),intermed2)
        
        #        print 'Press c to continue to plotting'
        #        pdb.set_trace()
        #Plot w_hat
        plt.subplot(2,ncycles,count+1)
        plt.plot(cb, nominal_SB, 'ks')
        plt.plot(cb, nominal_SB, '-k')
        plt.plot(cb, sbbin,'co')
        plt.plot(cb, result, 'bo')
        plt.plot(cb, wtilde, 'ro')
        ax = plt.gca()
        plt.xlim(.9*b[0], 1.1*b[-1])
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.subplot(2,ncycles,count+1+ncycles)
        plt.plot(cb, nominal_SB-result, 'bo')
        plt.plot(cb, nominal_SB-wtilde, 'ro')

    pdb.set_trace()    
    #Plot V
    plt.figure(2)
    plt.imshow(lhs, interpolation='nearest', cmap='Blues', origin = 'upper')
    pdb.set_trace()

    fig, ax = plt.subplots(figsize=(10,6))
    
    for draw in np.random.multivariate_normal(np.zeros_like(result), C, size=10):
        ax.plot(cb, draw, color="0.5")

    


    
if __name__ == "__main__":
    main()

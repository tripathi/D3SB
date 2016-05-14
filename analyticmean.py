import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.special as sc

def readinVis(datafile):
    """
    Extracts visibility data from a CASA and python generated .npz file.
    :param datafile: name of .npz file
    """
    datfile = np.load(datafile)
    return datfile['u'], datfile['v'], datfile['Vis'], datfile['Wgt']#, datfile['freq']   

    

def main():


    #Data input
    hiresvis = 'DATA/fullA.vis.npz'
    u, v, dvis, dwgt = readinVis(hiresvis)

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
    rleft = np.insert(initfile['b'][:-1],0, initfile['rin'])
    rright = initfile['b']
    N = rright.size


    #For ease of quick implementation, I'm coding up the function on paper, not the simplification Sean uses
    ## jarg = np.outer(2.*np.pi*rbin, rho)

#    jargleft = np.outer(rho, 2.*np.pi*rleft)
#    jargright = np.outer(rho, 2.*np.pi*rright)
#    jright = sc.j1(jargright)
#    jleft = sc.j1(jargleft)
#    x = (rright*sc.j1(jargright) - rleft*sc.j1(jargleft))
    
    x = np.empty([M,N])
    for j in range(M):
        for i in range(N): 
            x[j][i] = 1./rho[j]*(rright[i]*sc.j1(rho[j]*2.*np.pi*rright[i]) - rleft[i]*sc.j1(rho[j]*2.*np.pi*rleft[i]))
    pdb.set_trace()
    sigmainv = np.diag(dwgt.real)
    pdb.set_trace()
    test = np.dot(np.dot(np.transpose(x), sigmainv), x)
    pdb.set_trace()


    
        ## jinc = sc.j1(jarg)/jarg
    ## vrealnoshift = np.dot(2.*np.pi*rbin**2*intensity, jinc)

    ## x = 
    
if __name__ == "__main__":
    main()

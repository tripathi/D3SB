import numpy as np
import matplotlib.pyplot as plt
import pdb
from deprojectVis import deproject_vis
#We have Ndata data visibilities (D) that we'd like to solve for.
#The covariance on the data is Sigma
def main():
    visfilename = 'DATA/fullA_nf.vis.npz' ##Set
    datfile = np.load(visfilename)
    datfile['u'], datfile['v'], datfile['Vis'], datfile['Wgt']

    Dorig = datfile['Vis']
    uorig = datfile['u']
    vorig = datfile['v']
    rhoorig = np.sqrt(uorig**2. + vorig**2.)
    Dwgt = datfile['Wgt']
    #Sigmainv = np.diag(Dwgt) #a little unnecessary



    #Deproject the visibilities so that they're face-on
    #If binning, set these equal to the binned values
    incl = 50.
    PA = 70.
    offx = -0.3
    offy = -0.2

    nvisbins = 100 ##Set
    if (nvisbins>1):
        visbins = np.linspace(np.amin(rhoorig)/1000., np.amax(rhoorig)/1000., nvisbins)
        rhodeproj, Ddeproj, sigdeproj = deproject_vis([uorig, vorig, Dorig, Dwgt], visbins, incl, PA, offx, offy)
    else:
        rhodeproj, Ddeproj, sigdeproj = deproject_vis([uorig, vorig, Dorig, Dwgt], incl=incl, PA=PA, offx=offx, offy=offy)
    D = Ddeproj
    rho = rhodeproj
    Sigmainv = 1./np.square(np.absolute(sigdeproj))
    Ndata = D.size
    print 'Number of vis is', Ndata, np.shape(rho), np.shape(Sigmainv)

    #We have a model visibility (M) which uses Nbins annuli
    #M=Xw, where X is only a function of the bins and baseline

    #Select model annuli radii
    rmin = 0.01/140.
    rmax = 1.1
    nbins = 30
    radii = np.linspace(rmin, rmax, num=nbins+1) #Currently does NOT use rin
    rleft = radii[:-1]
    rright = radii[1:]
    rcenter = (rleft+rright)/2.

    pdb.set_trace()
# X =
#
# #Calculate uniform prior mean and covariance matrix
# #The mean of the distribution with a uniform prior is wu, with covariance Cu
#
# Cu = np.dot(np.dot(X.T, Sigmainv), X)
# Cuinv = np.linalg.inv(Cu)
# wu = np.dot(Cuinv, np.dot(np.dot(x.T, Sigmainv), Ddeproj))
#
# #Calculate the GP covariance matrix (Cw) from the kernel (k), with mean muw
# #The mean of the distribution with this prior is wgh, with variance Cgp
#
# k
# muw
# Cw =
# Cwinv = np.linalg.inv(Cw)
#
# Cgp = np.linalg.inv(Cuinv+np.Cwinv)
# wgp = np.dot(Cgp,(np.dot(Cuinv, wu) + np.dot(Cwinv, muw)))
    return

if __name__ == "__main__":
    main()

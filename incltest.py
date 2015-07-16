import pdb
import numpy as np
import matplotlib.pyplot as plt
from deprojectVis import deprojectVis
from getVis import getVis
from scipy.integrate import quad
from d3sbModel import d3sbModel
from astropy.io import fits



def similarityint(r, rc, ftot, gam):
    '''Similarity soln integrand'''
    return ((2-gam)*ftot/(2*np.pi*rc**2)) * r*(r/rc)**(-gam)*np.exp(-(r/rc)**(2.-gam))


plt.interactive(True)

origvis = 'DATA/ss_i75_carma.vis.fits'
facevis = 'DATA/ss_i0_carma.vis.fits'

#Original data
dataorig = getVis(origvis) #Used for CARMA visibilities.
uorig, vorig, drealorig, dimagorig, dwgtorig = dataorig
rhoorig = np.sqrt(uorig**2 +vorig**2)

#Face on data
dataface = getVis(facevis) #Used for CARMA visibilities.
uface, vface, drealface, dimagface, dwgtface = dataface
rhoface = np.sqrt(uface**2 + vface**2)
rind = rhoface.argsort()

#Data deprojected with deprojectVis
bins = np.arange(5.,500.,20)
visout = deprojectVis(origvis, bins, incl=75, PA = 0, offset=[0.,0.], nu = 225e9, wsc = 1., fitsfile=1)
newbins, newre, newim, newerr, Ruvp, realp = visout

#Data that I deprojected
incl = np.deg2rad(75.)
udeproj = uorig * np.cos(incl) #Deproject
rhodeproj = np.sqrt(udeproj**2+vorig**2)

#Binned truth
nbins = 20
rin = 0.01/140.
b = np.linspace(0.05, 1., num=nbins)
a = np.roll(b, 1)
a[0] = rin
cb = 0.5*(a+b)
rout = 100./140.
Ftot = 0.2
gam = 0.
wtrueint = np.zeros(nbins)
wtruebin = np.zeros(nbins)
for ibin in np.arange(nbins):
    wtrueint[ibin] = 2.*np.pi*quad(similarityint, a[ibin], b[ibin], args=(rout, Ftot, gam))[0]
    wtruebin[ibin] = wtrueint[ibin]/ np.pi/(b[ibin]**2.-a[ibin]**2.)

#Truth Visibilities
dbins = rin, b
uvsamples = uorig, vorig
p0 = np.concatenate(([incl], wtruebin))
vistransf = d3sbModel(p0, uvsamples, dbins)
p02 = np.concatenate(([0.], wtruebin))
vistransf2 = d3sbModel(p02, uvsamples, dbins)


## #Let's look at the surface brightness
## facesb = 'DATA/ss_i0_carma.fits' #The face-on profile
## origsb = 'DATA/ss_i75_carma.fits' #The inclined surface brightness profile

## fopen = fits.open(facesb)
## fopen.info()#Check its type
## scidata = np.squeeze(fopen[0].data)#Save the dimension of interest and remove extra singleton dimensions
## plt.imshow(scidata,cmap='gray', origin='lower')
    



plt.plot(rhoface[rind], drealface[rind], 'k')
plt.plot(rhoorig, drealorig, '.k', alpha=0.2)
#plt.plot(newbins, newre, 'bs')
#plt.plot(rhodeproj, drealorig, '.g')
#plt.plot(Ruvp, realp, '.m', alpha=0.1)
plt.plot(rhoorig, vistransf, '.r')
plt.plot(rhoorig, vistransf2, '.g')
pdb.set_trace()

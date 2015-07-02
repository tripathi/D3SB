import numpy as np
import matplotlib.pyplot as plt
#import triangle #Should work on tillandsia, but isn't
import pdb
from copy import deepcopy
from getVisALMA import getVisALMA
from lnprob import lnprob
from d3sbModel import d3sbModel
from scipy.integrate import quad
from scipy.optimize import curve_fit

def similaritysb(r, rc, ftot, gam):
    return ((2-gam)*ftot/(2*np.pi*rc**2)) * (r/rc)**(-gam)*np.exp(-(r/rc)**(2.-gam))

def similarityint(r, rc, ftot, gam):
    return ((2-gam)*ftot/(2*np.pi*rc**2)) * r*(r/rc)**(-gam)*np.exp(-(r/rc)**(2.-gam))

plt.interactive(True)


#Params
nbins=15
ndim, nwalkers = nbins, 4*nbins
plotchains = 1 #Plot individual chains or not
triangle = 0

#Input information
basename = 'blind1'
chainw0 = np.load('/Users/anjalitripathi/Downloads/mc_blind1_15_15bins_again_mean.npy')
#mc_blind1_20_abax2000_mean.npy')
hiresvis = basename + '.340GHz.vis.npz' #Model visibilities
data = getVisALMA('DATA/'+hiresvis)
infilecorr = np.load('opt_'+basename+'_linear_'+str(nbins)+'.npz')

samplesw0 = chainw0[:, 1000:, :].reshape((-1,nbins))

#Make triangle plot
if triangle:
    fig = triangle.corner(samplesw0)
    fig.savefig("linear_monot_"+basename+".png")
    pdb.set_trace()

#Get percentile information
stuff = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samplesw0, [16, 50, 84],axis=0)))
vcentral = zip(*stuff)[0]
vupper = zip(*stuff)[1]
vlower = zip(*stuff)[2]

if plotchains:
#Plot chains over time for each bin
    fig = plt.figure(1)
    for idim in np.arange(nbins):
        for iw in np.arange(nbins*4):
            plt.subplot(5,4,idim+1)
            plt.plot(chainw0[iw, :, idim], 'b')
            plt.plot([0,chainw0.shape[1]], [vcentral[idim], vcentral[idim]], 'k')
            plt.plot([0,chainw0.shape[1]], [vcentral[idim]-vlower[idim], vcentral[idim]-vlower[idim]], 'r')
            plt.plot([0,chainw0.shape[1]], [vcentral[idim]+vupper[idim], vcentral[idim]+vupper[idim]], 'g')




#Convert to appropriate units
dpc = 140. 



#Set bins
print 'Warning: Using hardcoded bins for Chi^2 calc'
#b = np.linspace(0.01, 1., num=nbins)
b = np.linspace(0.01, 1.5, num=nbins)


a = np.roll(b, 1)
rin = 0.01/dpc
a[0] = rin
cb = 0.5*(a+b)
dbins = rin, b


#Plot cumulative flux
ftest = np.pi*(b**2 - a**2)
fbin = np.asarray(vcentral)*np.pi*(b**2 - a**2)
fbinmean = infilecorr['w0']*np.pi*(b**2 - a**2)
cumf = np.cumsum(fbin)
cumfmean = np.cumsum(fbinmean)
plt.figure(10)
plt.plot(cb, cumf, '-ob', cb, cumfmean, '-.k')
## pdb.set_trace()
    
  

vcentralpos = np.asarray(vcentral)
vcentralpos[vcentralpos<0]=0

#Print chi^2 for each model
print 'Input mean guess', -2. * lnprob(infilecorr['w0'], data, dbins)
print 'Emcee output', -2. * lnprob(np.asarray(vcentral), data, dbins)
print 'Emcee central >=0', -2.*lnprob(np.asarray(vcentralpos), data, dbins)


fig6 = plt.figure(6)

herr = (b-a)/2.

#Truth, according to Sean    
rr = deepcopy(infilecorr['cb'])
rc = 75./dpc
gam = 0.3
ftot = 0.15
himage = np.zeros_like(rr)
himage[(rr>0)&(rr<rc)] = (rc/rr[(rr>0)&(rr<rc)])**(gam+0.5)
himage[(rr>rc)] = (rc/rr[(rr>rc)])**6.0
image = ftot*himage/np.sum(himage*np.pi*(b**2 - a**2))
plt.plot(rr, image, '-ks', alpha = 0.5)

plt.errorbar(infilecorr['cb'], vcentral, yerr = [vlower, vupper], xerr = herr, fmt='.b', elinewidth=1.5)
plt.plot(infilecorr['cb'], infilecorr['w0'], 'o', markeredgecolor='r', markerfacecolor='None', alpha=0.8)
print infilecorr['cb']


plt.title('15 bins to 1"')
plt.xlabel('Angle ["]')
plt.ylabel('Intensity [Jy/"$^2$]')

plt.xlim(2e-3,1.6)
plt.ylim(1e-3, 10)

ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')

#fig6.savefig(basename+"_"+str(nbins)+".png")
pdb.set_trace()



fig7 = plt.figure(7)
u, v, dreal, dimag, dwgt = data

rho = np.sqrt(u**2+v**2)
uvsamples = u, v
vis = d3sbModel(vcentral, uvsamples, dbins)
rind = rho.argsort()

plt.subplot(1,2,1)
plt.plot(rho, dreal, '.k')
plt.plot(rho[rind], vis[rind], 'b')
plt.ylabel('Visibility')
plt.xlabel('Rho')

plt.subplot(1,2,2)
plt.plot(rho, np.zeros_like(rho))
plt.plot(rho, dreal - vis, '.b')
plt.ylabel('Dreal - Model')
plt.xlabel('Rho')


#Fit central values to similarity solution to determine params
popt, pcov = curve_fit(similaritysb, infilecorr['cb'], vcentral, sigma=np.mean([vlower, vupper]), p0 = [.54, .15, .5], absolute_sigma=1)
print popt

print 'Stopping'
pdb.set_trace()



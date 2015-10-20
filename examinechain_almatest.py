import pdb
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.integrate import quad
from scipy.optimize import curve_fit
import triangle #Should work on tillandsia, but isn't
from getVis import getVis
from getVisALMA import getVisALMA
from lnprob_alone import lnprob_alone
#from d3sbModel import d3sbModel
from deprojectVis import deprojectVis

plt.style.use('mystyle')
#plt.interactive('True')

def inclfix(theta):
    incl = np.mod(theta, 360.)
    inclout = deepcopy(incl)
    #Rotate angle into 4 traditional quadrants
    #Inclination definition will need to change if profile not axisymm.
    q1 = np.where((incl>90)*(incl < 180))
    q2 = np.where((incl>=180)*(incl < 270))
    q3 = np.where(incl>=270)

    inclout[q1] = 180. - incl[q1]
    inclout[q2] = incl[q2] - 180.
    inclout[q3] = incl = 360.-incl[q3]
    inclout = inclout*np.pi/180.

    return inclout

def similaritysb(r, rc, ftot, gam):
    '''Similarity soln'''
    return ((2-gam)*ftot/(2*np.pi*rc**2)) * (r/rc)**(-gam)*np.exp(-(r/rc)**(2.-gam))

def similarityint(r, rc, ftot, gam):
    '''Similarity soln integrand'''
    return ((2-gam)*ftot/(2*np.pi*rc**2)) * r*(r/rc)**(-gam)*np.exp(-(r/rc)**(2.-gam))

def knee(r, rc, gam, A1, A2):
    '''Double power law'''
    out = np.zeros_like(r)
    out[r<rc] = A1*(rc/r[r<rc]**(gam+0.5))
    out[r>=rc] = A2*rc/r[r>=rc]**4.0
    return out


########
# MAIN #
########

#If running in ipython, don't comment this:
#(It eliminates the need for plt.show, etc)
#plt.interactive(True)

#Params
nbinsinit = 30
#20
#40
nbins=21
#14
#22
ndim = nbins
nwalkers = 4.*nbins
plotchains = 0#Plot individual chains or not
plottriangle = 0 #Make triangle plot or not
binmin = 0.01
#0.02
binmax = 1.6#2.0
#0.75

#Convert to appropriate units
dpc = 140.

#Find appropriate files
basename = 'gp_gap'
#test'
#blind2_fo'
note = 'gpl2'
#penalty1'
#gpcovarminus2'
#penalty1_no0inguess'
#gppriorminus2bin'

#penalty1'
#smalla_100k'
#penaltyrcoeff1'
#gp_test_fixedal'
#'withpenaltyforreal'
#
#smalla_100k'
#'binsmoved'
#'
#withnlog2a005'
#gpindices10k'
#choglobalfixedal'
#gaprcoeff_c_c_c_c'
#fracdivplus1'
#divpenalty'
#onlyturns_norcoeff'
#hyperlt025'
#hyper001'
#nopenalty'
#hyperlt025'
#reghyperp'
#big2bins_notmonot'

#monot'
#newinclball'
#a'
#vproj_longer'
#blindknee_i75_carma'
#'blind1'
chainw0 = np.load('mc_'+basename+'_'+str(nbins)+'_'+note+'_mean.npy')

#mc_blind2_20_uproj_mean.npy')
#mc_blind2_20_newinclball_mean.npy')
#mc_blind2_20_morebins_vproj_mean.npy')
#mc_blind2_fo_20_widerrange_mean.npy')
#mc_blind2_fo_12_fixedincl_mean.npy')
#mc_blind2_fo_15_inclstillinp0_mean.npy')
#mc_blind2_fo_12_a_mean.npy')

#mc_blind2_15_'+note+'_mean.npy')
#mc_blind2_15_incltest_vproj_mean.npy')
#mc_blind2_15_test_mean.npy')
#mc_blindknee_i75_carma_15_uproj50k_mean.npy')
#mc_blindknee_i75_carma_15_uproj10k_mean.npy')
#mc_blindknee_i75_carma_20_incltest_mean.npy')
#mc_blindknee_i75_carma_20_again_mean.npy')
#mc_blind1_20_abax2000_mean.npy')
hiresvis = 'DATA/'+basename+'.combo.noisy.vis.npz'
#.340GHz.vis.npz' #Model visibilities
#+'.vis.fits'
#
data = getVisALMA(hiresvis)
#ALMA('DATA/'+hiresvis)
infilecorr = np.load('opt_'+basename+'_linear_'+str(nbins)+'.npz')

#Fix inclinations
#chainw0[:,:,0] = inclfix(chainw0[:,:,0]).reshape(chainw0[:,:,0].shape)*180./np.pi

#Flatten chain
cstart = 8000
samplesw0 = chainw0[:, cstart:, :].reshape((-1,ndim))
sampleswonly = samplesw0
#sampleswonly = chainw0[:,cstart:, 2:].reshape((-1,ndim-2))

#Set bins
print 'Warning: Using hardcoded bins for Chi^2 calc'
btmp = np.linspace(binmin, binmax/2., num=nbinsinit/2)
btmp2 = np.linspace(binmax/2., binmax, num=nbinsinit/4)
b=np.concatenate([btmp, btmp2[1:]])

#btmp = np.linspace(binmin, binmax, num=nbinsinit)
#b = btmp
#np.concatenate([btmp[btmp<0.4], np.array([0.45, 0.6, binmax])])
#b = np.concatenate([btmp[btmp<0.35], np.array([0.35, 0.45, 0.6, binmax])])
#b = np.concatenate([btmp[btmp<0.5], np.array([0.6, binmax])])
numbins = np.shape(b) #Changing number of bins
nbins = numbins[0]
#b = np.linspace(0.01, 1.5, num=nbins)
a = np.roll(b, 1)
rin = 0.1/dpc
#0.01

a[0] = rin
cb = 0.5*(a+b)
dbins = rin, b
herr = (b-a)/2. #Bin extent (horiz. errorbar)

#Get percentile information - here 1 sigma
stuff = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samplesw0, [16, 50, 84],axis=0)))
vcentral = zip(*stuff)[0]
vupper = zip(*stuff)[1]
vlower = zip(*stuff)[2]

#In case average values are <0 (they shouldn't be)
#look at >=0 results
vcentralpos = np.asarray(vcentral)
vcentralpos[vcentralpos<0]=0



#Truth
rr = deepcopy(infilecorr['cb'])
himage = np.zeros_like(rr)

#gap
# rc = 50./dpc
# gam=0.5
# ftot=0.15
#
# himage[rr>0] = (rc/rr[rr>0])*np.exp(-(rr[rr>0]/rc)**(2.-gam))
# himage[(rr>25./dpc)*(rr<35./dpc)] *= 0.1


#blind1
#rc = 75./dpc
#gam = 0.3
#ftot = 0.15
#himage = np.zeros_like(rr)
#himage[(rr>0)&(rr<rc)] = (rc/rr[(rr>0)&(rr<rc)])**(gam+0.5)
#himage[(rr>rc)] = (rc/rr[(rr>rc)])**6.0

#blind2
#rc = 55./dpc
#gam = 0.5
#ftot = 0.12
#himage = np.zeros_like(rr)
#himage[(rr>0)&(rr<rc)] = (rc/rr[(rr>0)&(rr<rc)])**(gam+0.5)
#himage[(rr>rc)] = (rc/rr[(rr>rc)])**4.0
#image =0.0824975*himage
#ftot*himage/np.sum(himage)

#gp_test
#rc = 0.5
#ftot = 0.15
#arcsec = 206264.806427
tiny = 1e-12
#A = 0.10579 / arcsec**2
#himage[rr<=2.] =  A * (rr/rc)**(-1) * np.exp(-(rr/rc)**(1.5) )

#gp_gap
rc = 0.7
himage  =  (rr/rc)**(-0.75) * np.exp(-(rr/rc)**(2.5))
himage[(rr>0.4)&(rr<0.5)] *= 0.1
Ic = 0.062211   # exactly
himage *=Ic
himage[rr>2.] = 0.+tiny
image = himage
#ftot*himage/np.sum(himage*np.pi*(b**2 - a**2))

#print ftot/np.sum(himage*np.pi*(b**2 - a**2))

#knee(infilecorr['cb'], rc, gam, 1., 1.)
#0.0824975


#Print chi^2 for each model
gpa = .005
#.0001
gpl = 2.
#print 'Input mean guess (assuming 0 inclination)', -2. * lnprob_alone(infilecorr['w0'], data, dbins, gpa, gpl)
#print 'Emcee output', -2. * lnprob_alone(np.asarray(vcentral), data, dbins, gpa, gpl)
#print 'Truth', -2.*lnprob_alone(image, data, dbins, gpa, gpl)
#print 'Press c to continue to plotting stage'
#pdb.set_trace()

if plottriangle:
    #Make triangle plot
    fig = triangle.corner(sampleswonly)
    fig.savefig("linear_monot_"+basename+"_"+note+".png")
    pdb.set_trace()


if plotchains:
#Plot chains over time for each bin
    chainx = np.array([0,chainw0.shape[1]-cstart])/1000.
    chainxall = np.arange(chainw0.shape[1]-cstart)/1000.
    fig = plt.figure(1)

    for idim in np.arange(ndim):
        for iw in np.arange(ndim*4):
            plt.subplot(6,5,idim+1)
            plt.plot(chainxall, chainw0[iw,cstart:, idim], 'b')
            plt.plot(chainx, [vcentral[idim], vcentral[idim]], 'k')
            plt.plot(chainx, [vcentral[idim]-vlower[idim], vcentral[idim]-vlower[idim]], 'r')
            plt.plot(chainx, [vcentral[idim]+vupper[idim], vcentral[idim]+vupper[idim]], 'g')

    fig.savefig("chains_"+basename+"_"+note+".png")



# #Plot number of turning points
# fig = plt.figure(33)
# #turns = np.zeros((ndim*4, cstart))
# turns = np.zeros((ndim*4, chainw0.shape[1]))
# for iw in np.arange(ndim*4):
# #    for iter in np.arange(cstart):
#     for iter in np.arange(chainw0.shape[1]):
#         dx = np.diff(chainw0[iw,iter,1:])
#         turns[iw,iter] = np.sum(dx[1:]*dx[:-1] <0)
# #    totaliter = np.arange(cstart)
#     totaliter = np.arange(chainw0.shape[1])
# #    plt.subplot(2,1,1)
#     plt.plot(totaliter, turns[iw,:], '-or', alpha = 0.2)
# #    plt.plot(chainw0[iw,:,0], turns[iw,:], '-or', alpha = 0.2)
#     plt.title('Turning points vs step #')
# #penalty coefficient')
#     plt.ylim(-0.5,12.5)
# #    plt.subplot(2,1,2)
# #    plt.plot(totaliter, chainw0[iw,:,0], 'b')
# #    plt.title('Penalty coefficient')
# fig.savefig("turns_"+basename+"_"+note+".png")
# print 'Finished plotting turning points'
# pdb.set_trace()


#Plot cumulative flux

ftest = np.pi*(b**2 - a**2)
fbin = np.asarray(vcentral)*np.pi*(b**2 - a**2)
#fbin = np.asarray(vcentral)*np.pi*(b**2 - a**2)
fbinmean = infilecorr['w0']*np.pi*(b**2 - a**2)
cumf = np.cumsum(fbin)
cumfmean = np.cumsum(fbinmean)
fig10 = plt.figure(10)
ax = plt.gca()
ax.grid('off')

plt.plot(cb, cumf, '-o', cb, cumfmean, '-.k') #b
fig10.savefig("cumulative_"+basename+"_"+note+".png")


#Plot surface brightness
fig6 = plt.figure(6)
#for iw in np.arange(ndim*4):
    # plt.plot(infilecorr['cb'],chainw0[iw,0,], '-co', alpha = 0.1) #Plot starting ball
plt.plot(rr, image, '-ks', alpha = 0.4, linewidth=2) #Plot truth #k .5alpha
#plt.errorbar(infilecorr['cb'], vcentral, yerr = [vlower, vupper], xerr = herr, fmt='.b', elinewidth=1.5)
plt.errorbar(infilecorr['cb'], vcentral, yerr = [vlower, vupper],  fmt='.', elinewidth=2, c='#003399',markersize=12, markeredgewidth=1.) #b
plt.plot(infilecorr['cb'], infilecorr['w0'], 'o', markerfacecolor='None', markeredgewidth=1.,markeredgecolor='#fc4f30', zorder=1) #r


#Make it into a stairstep, thereby removing xerr=herr
for i in np.arange(nbins):
    binwidth = [a[i],b[i]]
    binheight=[vcentral[i], vcentral[i]]
    plt.plot(binwidth, binheight, c='#003399', linewidth=1.25, alpha=0.3)
    if i<nbins-1:
        plt.plot([b[i],b[i]], [vcentral[i],vcentral[i+1]], c='#003399', linewidth=1.25, alpha=0.3)

plt.title(str(nbins)+' bins') #Hardcoded
plt.xlabel('Angle ["]')
plt.ylabel('Intensity [Jy/"$^2$]')
plt.xlim(5e-4,2.0) #2e-3, 1.6
plt.ylim(1e-5, 30) #1e-3, 20
ax = plt.gca()
ax.grid('off')
ax.set_yscale('log')
ax.set_xscale('log')
fig6.savefig(basename+"_"+note+"_"+str(nbins)+".png")
pdb.set_trace()

#Plot visibilities & residuals
fig7 = plt.figure(7)
u, v, dreal, dimag, dwgt = data

newbins = np.arange(5., 2000., 50.)
visout = deprojectVis(hiresvis, newbins, incl=vcentral[0], PA=90., offset=[0., 0.], nu=340e9, wsc=1., fitsfile=0)
onewbins, obre, obim, ober, Ruvp, realp = visout

vproj = v*np.cos(vcentral[0]*(np.pi/180.))
#uprojt = u*np.cos(75.*np.pi/180.)
rho = np.sqrt(u**2+v**2)
rind = rho.argsort()
rhoproj = np.sqrt(vproj**2+u**2)
#rhoprojt = np.sqrt(v**2+uprojt**2.)

uvsamples = u, v
#uvsamplest = uprojt, v

pdb.set_trace()
vis = d3sbModel(vcentral, uvsamples, dbins)
# vist = d3sbModel(np.concatenate([[0],image]), uvsamples, dbins)

plt.subplot(2,1,1)
#plt.plot(rho, dreal, '.k')
#plt.plot(rhoproj, dreal, '.m', alpha = 0.1)
#plt.plot(Ruvp, realp, 'ys', alpha = 0.2)



plt.plot(rhoproj, dreal, '.m', alpha=0.2)
plt.plot(rho, vist, '.r', alpha = 0.2)
plt.plot(rho, vis, '.g', alpha = 0.2)

plt.plot(onewbins, obre, 'bs')
#plt.plot(rhoproj[rind], vis[rind], '.b')

plt.ylabel('Visibility')
plt.xlabel('Rho')


dv = dreal - vis
dvt = dreal - vist
dvrel = dv/dreal
dvtrel = dvt/dreal

toohigh = np.where(np.asarray(dvrel)>10000.)
#[i for i in dvrel if i>=10000]
toohight = np.where(np.asarray(dvtrel)>10000.)
print toohigh, dreal[toohigh], vis[toohigh], dvrel[toohigh]
print toohight, dreal[toohight], vist[toohight], dvtrel[toohight]


plt.subplot(2,1,2)
plt.plot(rho, np.zeros_like(rho), 'k')
plt.plot(rho, (dreal - vis), '.g')
plt.plot(rho, (dreal - vist), '.r', alpha = 0.5)
plt.ylabel('(Data-Model)')
plt.xlabel('Rho')
fig7.savefig("vis_"+basename+"_"+note+".png")
pdb.set_trace()

#Fit central values to determine params and recovery success
#popt, pcov = curve_fit(knee, infilecorr['cb'], vcentral, sigma=np.mean([vlower, vupper]), absolute_sigma=1)

#popt, pcov = curve_fit(similaritysb, infilecorr['cb'], vcentral, sigma=np.mean([vlower, vupper]), p0 = [.54, .15, .5], absolute_sigma=1)
#print popt

print 'Stopping'
pdb.set_trace()

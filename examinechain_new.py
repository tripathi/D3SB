import pdb
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.integrate import quad
from scipy.optimize import curve_fit
import corner #Should work on tillandsia, but isn't
import main as mf
import d3sbfxns as f
from discreteModel import discretemodel

plt.style.use('mystyle')

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
#plt.interactive(True)

#Parameters
basename = 'gaptest3'
#3'
#fullA'
#'fullB'
note ='gapcenter2'
#smallergap_powerlaw'
#t6'
#smallerdeeper'
#gap'
#gap_fullAsynthimg'
#longerarcsec2'
#'6k'
nbins = 22
#25
ndim = 26
dpc = 140.

plotchains = 1#Plot individual chains or not
plottriangle = 0 #Make triangle plot or not

#Files to load
infilecorr =np.load('init_'+basename+'_'+str(nbins)+'.npz')
chainw0 = np.load('mc0_'+basename+'_'+str(nbins)+'_'+note+'.npy')

## print 'Plotting turning points in initial ball'
## turns=np.zeros((nbins*4,chainw0.shape[1]))                                                                                                             
## fig = plt.figure(1)
## for iw in np.arange(nbins*4):
##     for iter in np.arange(chainw0.shape[1]):
##         dx = np.diff(chainw0[iw,iter,1:])
##         turns[iw,iter] = np.sum(dx[1:]*dx[:-1] <0)
##     totaliter =np.arange(chainw0.shape[1])
##     plt.plot(totaliter/1000, turns[iw,:], '-or', alpha = 0.2) 
##     plt.title('Turning points vs step ')                                                                                                                                 
## fig.savefig("turns_"+basename+"_"+note+".png") 
## plt.clf()


chainw1 = np.load('mc1_'+basename+'_'+str(nbins)+'_'+note+'.npy')

synthimg = 'DATA/'+basename+'.image.fits' #Synthesized image, for guesses     
datfile = np.load('DATA/'+basename+'.vis.npz')
data = datfile['u'], datfile['v'], datfile['Vis'], datfile['Wgt']   
    

#Flatten chain
cstart = 3000
samplesw1 = chainw1[:, cstart:, :].reshape((-1,ndim))
sampleswonly = chainw1[:,cstart:, 4:].reshape((-1,nbins))

#Get bins
cb = infilecorr['cb']
rin = infilecorr['rin']
b = infilecorr['b']
sbbin = infilecorr['sbbin']
a = np.roll(b, 1)
a[0] = rin
herr = (b-a)/2. #Bin extent (horiz. errorbar)

#dbins = rin, b


#Get percentile information - here 1 sigma
stuff = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samplesw1, [16, 50, 84],axis=0)))
vcentral = zip(*stuff)[0]
vupper = zip(*stuff)[1]
vlower = zip(*stuff)[2]

rsb, sb, beaminfo = f.sbdata(synthimg, vcentral[1], vcentral[0], vcentral[2], vcentral[3]) #Add plotting argument, if plot desired




## #Truth
## rr = deepcopy(infilecorr['cb'])
## himage = np.zeros_like(rr)


## #gp_test
## rc = 0.5
## #ftot = 0.15
## arcsec = 206264.806427
## tiny = 1e-12
## A = 0.10579 #/ arcsec**2
## himage[rr<=2.] =  A * (rr/rc)**(-1) * np.exp(-(rr/rc)**(1.5) )

## #gp_gap
## rc = 0.7
## himage  =  (rr/rc)**(-0.75) * np.exp(-(rr/rc)**(2.5))
## #himage[(rr>0.4)&(rr<0.5)] *= 0.1
## #Ic = 0.062211   # exactly
## Ic = 0.054976#For no gap

## himage *=Ic
## #himage[rr>2.] = 0.+tiny
## image = himage
## #ftot*himage/np.sum(himage*np.pi*(b**2 - a**2))

## #print ftot/np.sum(himage*np.pi*(b**2 - a**2))

## #knee(infilecorr['cb'], rc, gam, 1., 1.)
## #0.0824975


#Print chi^2 for each model
#print 'Input mean guess (assuming 0 inclination)', -2. * lnprob_alone(infilecorr['w0'], data, dbins, gpa, gpl)
#print 'Emcee output', -2. * lnprob_alone(np.asarray(vcentral), data, dbins, gpa, gpl)
#print 'Truth', -2.*lnprob_alone(image, data, dbins, gpa, gpl)
print 'Press c to continue to plotting stage'


if plottriangle:
    #Make triangle plot
    print "I'm making a triangle plot!"
    fig = corner.corner(sampleswonly)
    fig.savefig("corner_"+basename+"_"+note+".png")
    pdb.set_trace()


if plotchains:
#Plot chains over time for each bin
    print "I'm plotting chains!"
    chainx = np.array([0,chainw1.shape[1]-cstart])/1000.
    chainxall = np.arange(chainw1.shape[1]-cstart)/1000.
    fig = plt.figure(1)

    for idim in np.arange(ndim):
        for iw in np.arange(ndim*4):
            plt.subplot(6,5,idim+1)
            plt.plot(chainxall, chainw1[iw,cstart:, idim], 'b',linewidth=0.5)
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
print "Let me calculate cumulative flux"
ftest = np.pi*(b**2 - a**2)
fbin = np.asarray(vcentral[ndim-nbins:])*np.pi*(b**2 - a**2)
fbinmean = sbbin*np.pi*(b**2 - a**2)
cumf = np.cumsum(fbin)
cumfmean = np.cumsum(fbinmean)
fig10 = plt.figure(10)
ax = plt.gca()
ax.grid('off')

plt.plot(cb, cumf, '-o', cb, cumfmean, '-.k') #b
fig10.savefig("cumulative_"+basename+"_"+note+".png")


#Plot surface brightness
fig6 = plt.figure(6)
plt.clf()
plt.plot(rsb, sb, '.y', alpha = 0.1)

# Sean's "true" SB distribution    
#flux = 0.12
#sig = 0.7
#incl = 35.
#PA = 140.
#offx = 0.0
#offy = 0.2
#SB = (sig/cb)**0.9 * np.exp(-(cb/sig)**3.5)



inbins = 2000
ib = 0.001 + 0.001*np.arange(inbins)
ia = np.roll(ib, 1)
ia[0] = 0.1/140.
ir = 0.5*(ia+ib)

iflux = 0.12
isig = 0.6#7
nominal_SB = (ir/isig)**-0.7 * np.exp(-(ir/isig)**2.5)
#SB = (ir/isig)**-0.9 * np.exp(-(ir/isig)**3.5)
int_SB = np.sum(np.pi*nominal_SB*(ib**2-ia**2))
nominal_SB *= iflux/int_SB

#add in a resolved gap
gapcenter = 30./140.
gapwidth = 0.03
gapdepth = 0.85
# determine amplitude of gaussian
func = interpolate.interp1d(ir, nominal_SB)
gap_amp = gapdepth*func(gapcenter)
gap_profile = gap_amp * np.exp(-0.5*(ir-gapcenter)**2/gapwidth**2)
SB = nominal_SB - gap_profile

plt.plot(ir, SB, '-k')

for iw in np.arange(ndim*4):
    plt.plot(cb,chainw1[iw,0,ndim-nbins:], 'oc', alpha = 0.05) #Plot starting ball , linewidth=1
#plt.plot(rr, image, '-ks', alpha = 0.4)#, linewidth=2) #Plot truth #k .5alpha
plt.errorbar(cb, vcentral[ndim-nbins:], yerr = [vlower[ndim-nbins:], vupper[ndim-nbins:]],xerr=herr,  fmt='ob', elinewidth=2, alpha= 0.8) #b #003399 c='#30a2da',markersize=12
plt.plot(cb, sbbin, 'o', markerfacecolor='None', markeredgewidth=1.,markeredgecolor='#fc4f30', alpha = 0.7) #r , zorder=1




#Make it into a stairstep, thereby removing xerr=herr
#for i in np.arange(nbins):
#    binwidth = [a[i],b[i]]
#    binheight=[vcentral[i], vcentral[i]]
#    plt.plot(binwidth, binheight, c='#30a2da', alpha=0.9)
##003399', linewidth=1.25, alpha=0.3)
#    if i<nbins-1:
#        plt.plot([b[i],b[i]], [vcentral[i],vcentral[i+1]], c='#30a2da', alpha=0.9)#, c='#003399', linewidth=1.25, alpha=0.3)

#plt.title(str(nbins)+' bins') #Hardcoded
plt.xlabel('Angle ["]')
plt.ylabel('Intensity [Jy/"$^2$]')
plt.xlim(1e-3,3.0) #2e-3, 1.6
plt.ylim(1e-5, 10) #1e-3, 20
ax = plt.gca()
ax.grid('off')
ax.set_yscale('log')
ax.set_xscale('log')
fig6.savefig("sb_"+basename+"_"+note+"_"+str(nbins)+".png")
pdb.set_trace()

pdb.set_trace()

#Plot visibilities & residuals
fig7 = plt.figure(7)
u, v, dvis, dwgt = data


newbins1 = np.arange(1., 252., 50.)
newbins2 = np.arange(251., 2400., 200.)
newbins = np.concatenate([newbins1, newbins2[1:]])

#visout = deprojectVis(hiresvis, newbins, incl=vcentral[0], PA=90., offset=[0., 0.], nu=340e9, wsc=1., fitsfile=0)
#onewbins, obre, obim, ober, Ruvp, realp = visout

#vproj = v*np.cos(vcentral[0]*(np.pi/180.))
#uprojt = u*np.cos(75.*np.pi/180.)
rho = np.sqrt(u**2+v**2)
rind = rho.argsort()
rhoproj = rho
print 'WARNING: Assuming 0 inclination for visibility'
#np.sqrt(uproj**2+v**2)
#rhoprojt = np.sqrt(v**2+uprojt**2.)

uvsamples = u, v
#uvsamplest = uprojt, v


vis = discretemodel(vcentral, uvsamples, rin, b, fitproj=1)
#vist = discretemodel(image, uvsamples, dbins)
# vist = d3sbModel(np.concatenate([[0],image]), uvsamples, dbins)

plt.subplot(2,1,1)
#plt.plot(rho, dreal, '.k')
#plt.plot(rhoproj, dreal, '.m', alpha = 0.1)
#plt.plot(Ruvp, realp, 'ys', alpha = 0.2)



plt.plot(rhoproj, dvis.real, '.', alpha=0.2, color='#FFB5B8')

#plt.plot(rho, vist, '.k', alpha = 0.1)
plt.plot(rho, vis, '.b', alpha = 0.1)
#plt.plot(onewbins, obre, '.m', alpha = 0.7, markersize = 10)

#plt.plot(rhoproj[rind], vis[rind], '.b')

plt.ylabel('Visibility')
plt.xlabel('Rho')
plt.show()

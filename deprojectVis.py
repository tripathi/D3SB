import numpy as np
from astropy.io import fits
import pdb

def deprojectVis(file, bins, incl=0., PA=0., offset=[0., 0.], nu=1e12, wsc=1, fitsfile=0.):

    # - read in, parse data
    # - Read from FITS file or .npz, as appropriate
    if (fitsfile):
        dvis  = fits.open(file)
        u = 1e-3*dvis[0].data.par(0)[:]*nu#/2.9974e8
        v = 1e-3*dvis[0].data.par(1)[:]*nu#/2.9974e8
        real = np.squeeze(dvis[0].data['Data'])[:,0]
        imag = np.squeeze(dvis[0].data['Data'])[:,1]
        wgt  = np.squeeze(dvis[0].data['Data'])[:,2]
        dvis.close()
    else:
        vin = np.load(file)
        freq = nu
        u = 1e-3*vin['u']*freq/2.9974e8
        v = 1e-3*vin['v']*freq/2.9974e8
        real = vin['Re']
        imag = vin['Im']
        wgt = vin['Wt']

    # - polar coordinates
    rho   = np.sqrt(u**2 + v**2)
    theta = np.arctan2(v, u)

    # - amps and phases
    amp  = np.sqrt(real**2 + imag**2)
    pha  = np.arctan2(imag, real)

    # - resolve quadrant ambiguities
    pha[real<0] += 2.*np.pi
    theta[v<0] += 2.*np.pi

    # - reweight
    wgt *= wsc

    # - de-project, rotate coordinates
    dmaj = rho*np.cos(theta-(np.radians(180.-PA)))*np.cos(np.radians(incl))
    dmin = rho*np.sin(theta-(np.radians(180.-PA)))
    Ruv = np.sqrt(dmaj**2+dmin**2)

    # - shifts from phase center
    dra  = -np.radians(offset[0]/3600.)
    ddec = -np.radians(offset[1]/3600.)
    real = amp*np.cos(pha+2.*np.pi*1e3*(u*dra+v*ddec))
    imag = amp*np.sin(pha+2.*np.pi*1e3*(u*dra+v*ddec))

    # - toss any flagged data
    Ruv  = Ruv[wgt>0]
    real = real[wgt>0]
    imag = imag[wgt>0]
    wgt  = wgt[wgt>0]

    # - azimuthal averaging
    bwid = 0.5*(bins[1]-bins[0])
    bmin = bins-bwid
    bmax = bins+bwid
    nbin = np.size(bins)
    b_re = np.zeros_like(bins)
    b_im = np.zeros_like(bins)
    b_er = np.zeros_like(bins)
    b_in = np.zeros_like(bins)

    for i in range(nbin):
        if ((Ruv > bmin[i]) & (Ruv <= bmax[i])).any():
            re_in = real[(Ruv > bmin[i]) & (Ruv <= bmax[i])]
            im_in = imag[(Ruv > bmin[i]) & (Ruv <= bmax[i])]
            wt_in = wgt[(Ruv > bmin[i]) & (Ruv <= bmax[i])]
            #print(np.std(re_in)/np.mean(1./np.sqrt(wt_in)))
            b_re[i], b_wt = np.average(re_in, weights=wt_in, returned=True)
            b_im[i] = np.average(im_in, weights=wt_in)
            b_er[i] = np.sqrt(1./b_wt)
            b_in[i] = np.size(re_in)

    # - remove bins with no data
    bins = bins[b_in > 0]
    b_re = b_re[b_in > 0]
    b_im = b_im[b_in > 0]
    b_er = b_er[b_in > 0]

    # - package results for return
    output = bins, b_re, b_im, b_er, Ruv, real
    return output

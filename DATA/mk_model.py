import numpy as np
import os
from astropy.io import fits

def mk_model(p, oname='model', dpc=140., ra_pc=165.466277958, dec_pc=-34.7047311944, nu=100., dnu=4):

    # - constants
    AU = 1.4960e13		# - definition of an AU [cm]
    pc = 3.0857e18		# - definition of a parsec [cm]
    rwid = 400*AU		# - 1/2 width of image (in cm)

    # - set free parameters
    ftot   = p[0]		# - total flux (Jy)
    rc     = p[1]*AU
    gam    = p[2]
    incl   = p[3]
    PA     = p[4]
    offs   = [0.0, 0.0]

    # - radial grid
    nx = 801		    # - preferrably an odd number
    ny = 801		    # - best if this is square
    xps = 2.*rwid/(nx-1)
    yps = 2.*rwid/(ny-1)
    xp_,yp_ = np.meshgrid(xps*(np.arange(nx)-(nx/2.-0.5)), \
                          yps*(np.arange(ny)-(ny/2.-0.5)))
    ang = (270.-PA)*np.pi/180.
    xp = np.cos(ang)*xp_-np.sin(ang)*yp_
    yp = np.sin(ang)*xp_+np.cos(ang)*yp_
    r = np.sqrt(xp*xp+1./np.cos(incl*np.pi/180.)**2*yp*yp)


    # - create image

    # similarity solution or power-law+cutoff
    himage = 0.*r
    # base structure (shallow inner disk, steep outer disk)
    # CHANGE THIS AS YOU LIKE
    himage[(r>0)&(r<rc)] = (rc/r[(r>0)&(r<rc)])**(gam+0.5)
    himage[(r>rc)&(r<(0.8*rwid))] = (rc/r[(r>rc)&(r<(0.8*rwid))])**6.0

    # normalize flux scale
    image = ftot*himage/np.sum(himage)

    # - construct header
    hdr = fits.Header()
    hdr.set('BITPIX',-32)
    hdr.set('CDELT1',-1.*xps/(AU*dpc)/3600.)
    hdr.set('CRPIX1',nx/2.+0.5)
    hdr.set('CRVAL1',ra_pc+offs[0]/np.cos(dec_pc*np.pi/180)/3600.)
    hdr.set('CTYPE1',  'RA---SIN')
    hdr.set('CDELT2',yps/(AU*dpc)/3600.)
    hdr.set('CRPIX2',ny/2.+0.5)
    hdr.set('CRVAL2',dec_pc+offs[1]/3600.)
    hdr.set('CTYPE2','DEC--SIN')
    hdr.set('CTYPE4','FREQ    ')
    hdr.set('CRPIX4',1)
    hdr.set('CDELT4',dnu*1e9)
    hdr.set('CRVAL4',nu*1e9)
    hdr.set('CTYPE3','STOKES  ')
    hdr.set('CRVAL3',1.)
    hdr.set('CDELT3',1.)
    hdr.set('CRPIX3',1.)
    hdr.set('EPOCH' ,2000)
  
    # - parse image
    fimage = np.reshape(np.float32(image),(1,1,ny,nx))

    # - export to fits
    hdu = fits.PrimaryHDU(fimage, header=hdr)
    hdulist = fits.HDUList([hdu])
    os.system('rm -rf '+oname+'.fits')
    hdulist.writeto(oname+'.fits', clobber=True)
    hdulist.close()

    return fimage

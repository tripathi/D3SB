import numpy as np
from astropy.io import fits

def getVisALMA(datafile):

   # Tool for extracting visibility data from a CASA and python generated .npz file.
   
    datain = np.load(datafile)
    freq = 340e9
    u = 1e-3*datain['u']*freq/2.9974e8
    v = 1e-3*datain['v']*freq/2.9974e8
    dreal = datain['Re']
    dimag = datain['Im']
    dwgt = 4000.*datain['Wt']

    data = u, v, dreal, dimag, dwgt

    return data




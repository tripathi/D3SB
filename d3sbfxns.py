import numpy as np

def makebins(rin, b):
    """
    Return bin centers for a given choice of b and rin
    """
    a = np.roll(b, 1)
    a[0] = rin
    cb = 0.5*(a+b)
    return cb
    
def 

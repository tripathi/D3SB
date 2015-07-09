from genmodelfits import genmodelfits
from mk_model import mk_model
import argparse
import sys

# Input
parser = argparse.ArgumentParser(description="Set array")
parser.add_argument('-a', metavar='INFILE', type=str, default='carma', help='Which array? carma? alma?')
args = parser.parse_args()


#Set input parameters
#Rc [AU], Total flux density [Jy], Gamma, Inclination [deg], Position Angle [deg], Distance [pc]
rc = 75.
ftot = 0.15
gamma = 0.3
incl = 75.
posangle = 0
dsource = 140.
base = 'blindknee_i75'
basename = base+'_'+args.a.lower()


#Create model image
if args.a.lower() == 'alma':
    print 'Making model ALMA image'
    pars = [ftot, rc, gamma, incl, posangle]
    foo = mk_model(pars, oname=basename, dpc=dsource)
elif args.a.lower() == 'carma':
    print 'Making model CARMA image'
    p = [rc, ftot, gamma, incl, posangle, dsource]
    foo = genmodelfits(p, basename)
else:
    print "Array unknown. Please retry."
    sys.exit()
    
#Write parameters to log file
f = open('models.log', 'a')
f.write(str(rc)+', '+str(ftot)+', '+str(gamma)+', '+str(incl)+', '+str(posangle)+', '+str(dsource)+', '+args.a.upper()+', '+basename+'\n')
f.close()


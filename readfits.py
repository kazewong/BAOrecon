import numpy as np
from astropy.io import fits
from reconFunction import*
from scipy.integrate import quad

def convert(z):
	return 299792.458/((np.sqrt(0.3089*(1+np.power(z,3))+0.6911)*67.74)*0.6774)

directory = '../data/'
cata = fits.open(directory+'galaxy_DR10v8_LOWZ_North.fits')
table = cata[1].data
RA = table.field('RA')
DEC = table.field('DEC')
Z = table.field('Z')
Density = table.field('nz')
for i in range(Z.size):
	Z[i] = quad(convert,0,Z[i])[0]
Cart = coordinateConvert(RA,DEC,Z)
readout = np.array([Cart[0],Cart[1],Cart[2],Density]).transpose()
np.savetxt(directory+'lowzNdata.dat',readout)

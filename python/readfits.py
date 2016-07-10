from astropy.io import fits
from scipy.integrate import quad
import numpy as np
import argparse
def convert(z):
	return 299792.45*0.7/((np.sqrt(0.3089*np.power(1+z,3)+0.6911))*67.6)

def coordinateTransform(z,RA,Dec):
	output = np.array([[0. for i in range(z.size)] for i in range(3)])
	output[0] = z*np.sin(Dec)*np.cos(RA)
	output[1] = z*np.sin(Dec)*np.sin(RA)
	output[2] = z*np.cos(Dec)
	return output

parser = argparse.ArgumentParser(description='Reconstructing galaxy catalog')
parser.add_argument('-data',help='Data file directory')
parser.add_argument('-output',default='./',help='The output file directory')
args=parser.parse_args()

file = np.genfromtxt(args.data).T
RA,Dec,Z = np.radians(file[0]),np.radians(90-file[1]),file[2]
for i in range(Z.size):
	Z[i]=quad(convert,0,Z[i])[0]
Cart = coordinateTransform(Z,RA,Dec)
readout = np.array([Cart[0],Cart[1],Cart[2],[1. for i in range(Cart[0].size)]]).T
np.savetxt(args.output,readout)

'''
directory = '/physics/wangkeiw/'
cata = fits.open(directory+'random2_DR10v8_CMASS_North.fits')
table = cata[1].data
print table.field('RA')[-1],table.field('DEC')[-1]
RA = np.radians(table.field('RA'))
Dec = np.radians(90-table.field('DEC'))
Z = table.field('Z')
RA = RA[(Z>0)*(Z<2)]
Dec = Dec[(Z>0)*(Z<2)]
Z = Z[(Z>0)*(Z<2)]
Density = table.field('nz')[(Z>0)*(Z<2)]
for i in range(Z.size):
	Z[i] = quad(convert,0,Z[i])[0]
Cart = coordinateTransform(Z,RA,Dec)
readout = np.array([Cart[0],Cart[1],Cart[2],[1. for i in range(Cart[0].size)]]).transpose()
np.savetxt(directory+'CMASSNrandom2.dat',readout)'''

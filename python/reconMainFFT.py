#from astropy.io import fits
from reconFunction import *
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Reconstructing galaxy catalog')
parser.add_argument('-data',help='Data file directory')
parser.add_argument('-rand',help='Random file directory')
parser.add_argument('-output',default='./',help='The output file directory')
parser.add_argument('-s',default=0,help='Smoothing scale')
parser.add_argument('-b',default=1,help='Clustering bias')
parser.add_argument('-f',default=0,help='Growth Factor')
args=parser.parse_args()
print args

gridNumber = 512
'''filename = 'CMASSN'
args.s = 0
directory = '/physics/wangkeiw/'
writeDirectory='/physics2/wangkeiw/'
args.b=1
args.f=0
'''
print 'Grid Number is '+str(gridNumber)
print 'Smoothing scale is '+str(args.s)
print 'Galaxy clustering bias is '+str(args.b)
print 'Growth Factor is '+str(args.f)

data = np.genfromtxt(args.data).T
random = np.genfromtxt(args.rand).T
dataCart = np.array([data[0],data[1],data[2]])
dataDensity = data[3]
randomCart = np.array([random[0],random[1],random[2]])
randomDensity = random[3]
dataCart = dataCart.transpose()
randomCart = randomCart.transpose()
#print randomCart.shape

boxmin = randomCart.min()
boxsize = randomCart.max()-boxmin
print 'The boxsize is '+str(boxsize)
print 'Start computing densityfield'
dataDensityField = computeDensityField(dataCart,dataDensity,gridNumber,boxsize,boxmin)
print 'The maximum of density field is ' + str(dataDensityField.max())
randomDensityField = computeDensityField(randomCart,randomDensity,gridNumber,boxsize,boxmin)
print 'The minimum of density field is ' + str(randomDensityField.max())
print 'Start masking densityField'
dataDensityField = maskDensityField(dataDensityField,randomDensityField)
np.savetxt('density1.dat',np.sum(dataDensityField.real,axis=(0,1)))
np.savetxt('density2.dat',np.sum(dataDensityField.real,axis=(1,2)))
np.savetxt('density3.dat',np.sum(dataDensityField.real,axis=(0,2)))

print 'The maximum of overdensity field is ' + str(dataDensityField.max())	
wavenumber = getWavenumber(gridNumber,boxsize)
print 'The pre-smoothing mean density is'+str(np.mean(dataDensityField))
print 'Start smoothing density field'
fft = np.fft.fftn(dataDensityField)
#fft = gaussianFilter(fft,float(args.s),wavenumber)
#fft[0][0][0]=0
print 'The post-smoothing mean density is'+str(np.mean(np.fft.ifftn(fft)))
print 'start computing Displacement'
displacementfft = computeDisplacement(fft,wavenumber,boxsize,float(args.b))
np.savetxt('dx1.dat',np.sum(displacementfft[0].real,axis=(0,1)))
np.savetxt('dy1.dat',np.sum(displacementfft[1].real,axis=(0,1)))
np.savetxt('dz1.dat',np.sum(displacementfft[2].real,axis=(0,1)))
np.savetxt('dx2.dat',np.sum(displacementfft[0].real,axis=(0,2)))
np.savetxt('dy2.dat',np.sum(displacementfft[1].real,axis=(0,2)))
np.savetxt('dz2.dat',np.sum(displacementfft[2].real,axis=(0,2)))
np.savetxt('dx3.dat',np.sum(displacementfft[0].real,axis=(1,2)))
np.savetxt('dy3.dat',np.sum(displacementfft[1].real,axis=(1,2)))
np.savetxt('dz3.dat',np.sum(displacementfft[2].real,axis=(1,2)))
redshiftDistortion = getRadialFactor(gridNumber,boxsize,boxmin)
displacementfftCorrected = displacementfft#-args.f*np.linalg.norm(displacementfft*redshiftDistortion,axis=0)*redshiftDistortion/(1+args.f)
print 'start shifting particles'
dataNew = shiftParticle(dataCart.T,displacementfftCorrected,gridNumber,boxsize,boxmin).T
randomNew = shiftParticle(randomCart.T,displacementfftCorrected,gridNumber,boxsize,boxmin).T
dataCart = dataCart.T
randomCart = randomCart.T
print 'Start writing data into output file'
datapostrecon = np.array([dataNew[0],dataNew[1],dataNew[2],np.array([1.0 for i in range(dataNew[0].size)])]).transpose()
randompostrecon = np.array([randomNew[0],randomNew[1],randomNew[2],np.array([1.0 for i in range(randomNew[0].size)])]).transpose()
np.savetxt(args.output+'.data',datapostrecon)
np.savetxt(args.output+'.rand',randompostrecon)

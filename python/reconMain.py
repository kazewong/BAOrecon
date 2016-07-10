from astropy.io import fits
import numpy as np
from reconFunction import *

gridNumber = 512
filename = 'CMASSN'
smoothingScale = 10
directory = '/physics/wangkeiw/'
bias=1


data = np.genfromtxt(directory+filename+'data.dat.txt').transpose()
random = np.genfromtxt(directory+filename+'random1.dat').transpose()
dataCart = np.array([data[0],data[1],data[2]])
dataDensity = data[3]
randomCart = np.array([random[0],random[1],random[2]])
randomDensity = random[3]
dataCart = dataCart.transpose()
randomCart = randomCart.transpose()
#print randomCart.shape

del data
del random

boxsize = dataCart.max()-dataCart.min()#getBoxSize(dataCart)
print 'Start computing densityfield'
dataDensityField = computeDensityField(dataCart,dataDensity,gridNumber,boxsize)
#print dataDensityField
randomDensityField = computeDensityField(randomCart,randomDensity,gridNumber,boxsize)
dataDensityField = maskDensityField(dataDensityField,randomDensityField)
print dataDensityField
wavenumber = getWavenumber(gridNumber,boxsize)
print 'smoothing density field'
fft = np.fft.fftn(dataDensityField)
fft = gaussianFilter(fft,smoothingScale,wavenumber)
smoothedField = np.fft.ifftn(fft)
print 'start computing Displacement'
displacement =displacementSolver(smoothedField,randomDensityField,gridNumber)
print 'start shifting particles'
dataNew = shiftParticle(dataCart.T,displacement,gridNumber,boxsize).T
randomNew = shiftParticle(randomCart.T,displacement,gridNumber,boxsize).T
dataCart = dataCart.T
randomCart = randomCart.T
datapostrecon = np.array([dataNew[0],dataNew[1],dataNew[2],np.array([1.0 for i in range(dataNew[0].size)])]).transpose()
randompostrecon = np.array([randomNew[0],randomNew[1],randomNew[2],np.array([1.0 for i in range(randomNew[0].size)])]).transpose()
np.savetxt(directory+filename+'datapostrecon'+str(smoothingScale)+'.dat',datapostrecon)
np.savetxt(directory+filename+'randompostrecon'+str(smoothingScale)+'.dat',randompostrecon)

from astropy.io import fits
import numpy as np
from reconFunction import *

gridNumber = 100
filename = 'lowzS'
smoothingScale = 20
directory = '../data/'

data = np.genfromtxt(directory+filename+'data.dat').transpose()
random = np.genfromtxt(directory+filename+'random.dat').transpose()
dataCart = np.array([data[0],data[1],data[2]])
dataDensity = data[3]
randomCart = np.array([random[0],random[1],random[2]])
randomDensity = random[3]
dataCart = dataCart.transpose()
randomCart = randomCart.transpose()
print randomCart.shape
print 'Start computing densityfield'
dataDensityField = computeDensityField(dataCart,dataDensity,gridNumber)
randomCart = generatingMask(dataDensityField,10,gridNumber,randomCart,dataCart.max())
#randomDensityField = computeDensityField(randomCart,randomDensity,gridNumber,dataCart.max())
#randomDensityField = mask*randomDensityField
fft = np.fft.fftn(dataDensityField)
wavenumber = getWavenumber(gridNumber,dataCart.max())
fft = gaussianFilter(fft,smoothingScale,wavenumber)
print 'start computing Displacement'
displacement = computeDisplacement(fft,wavenumber,dataCart.max())
displacement = displacement.transpose()
print 'start shifting particles'
dataNew = shiftParticle(dataCart,displacement,gridNumber)
dataNew = dataNew.transpose()
randomNew = shiftParticle(randomCart,displacement,gridNumber)
randomNew = randomNew.transpose()
dataCart = dataCart.T
randomCart = randomCart.T
datapostrecon = np.array([dataNew[0],dataNew[1],dataNew[2],np.array([1.0 for i in range(dataNew[0].size)])]).transpose()
randompostrecon = np.array([randomNew[0],randomNew[1],randomNew[2],np.array([1.0 for i in range(randomNew[0].size)])]).transpose()
np.savetxt(directory+filename+'randomprerecon.dat',randomprerecon)
np.savetxt(directory+filename+'randompostrecon'+str(smoothingScale)+'.dat',randompostrecon)

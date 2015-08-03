from astropy.io import fits
import numpy as np
from reconFunction import *

gridNumber = 100
filename = 'lowzS'
smoothingScale = 200
directory = '../data/'

data = np.genfromtxt(directory+filename+'data.dat').transpose()
dataCart = np.array([data[0],data[1],data[2]])
dataDensity = data[3]
dataCart = dataCart.transpose()
print 'Start computing densityfield'
dataDensityField = computeDensityField(dataCart,dataDensity,gridNumber)
mask = generatingMask(dataDensityField,100,gridNumber)
mean = dataDensityField[np.where(mask>0)].mean()
dataDensityField[np.where(mask==0)] = mean
dataDensityField = dataDensityField/mean - 1
fft = np.fft.fftn(dataDensityField)
wavenumber = getWavenumber(gridNumber,dataCart.max()-dataCart.min())
fft = gaussianFilter(fft,smoothingScale,wavenumber)
dataDensityField = np.fft.ifftn(fft)
dataDensityField[np.where(mask==0)] = 0
fft = np.fft.fftn(dataDensityField)
print 'start computing Displacement'
displacement = computeDisplacement(fft,wavenumber,dataCart.max()-dataCart.min())
displacement = displacement.transpose()
print 'start shifting particles'
dataNew = shiftParticle(dataCart,displacement,gridNumber)
dataNew = dataNew.transpose()
dataCart = dataCart.T
densityBefore = dataDensityField.flatten()
density = np.fft.ifftn(fft).flatten()
fs = fft.flatten().real
dx = np.fft.ifftn(displacement[0]).flatten().real
dy = np.fft.ifftn(displacement[1]).flatten().real
dz = np.fft.ifftn(displacement[2]).flatten().real
dm = np.sqrt(dx*dx+dy*dy+dz*dz)
i = j = k = loop = 0
output = np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] for local in range(dx.size)])
while i<displacement[0].shape[0]:
	s = np.sqrt(i*i+j*j+k*k)
	output[loop] = np.array([i+1,j+1,k+1,s,fs[loop],dx[loop],dy[loop],dz[loop],dm[loop],densityBefore[loop],density[loop]])
	k = k+1
	loop = loop +1
	if int(k) ==displacement[0].shape[0]:
		k = 0
		j = j+1
	if int(j) ==displacement[0].shape[0]:
		j = 0
		i = i+1
		print i
np.savetxt(directory+filename+'debug.dat',output)

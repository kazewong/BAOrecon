import numpy as np
from scipy import interpolate
from scipy.spatial import distance
def computeDensityField(position,density,gridNumber):
	axis = np.array([0. for i in range(gridNumber)])
	field = np.array(np.meshgrid(axis,axis,axis)[0])
	position = position.T
	index = position.copy()
	index[0] = np.floor((position[0]-position[0].min())*(gridNumber-1)/(2242-position[0].min()))
	index[1] = np.floor((position[1]-position[1].min())*(gridNumber-1)/(1425-position[1].min()))
	index[2] = np.floor((position[2]-position[2].min())*(gridNumber-1)/(1128-position[2].min()))
	position = position.T
	index = index.T
	print density.size,index
	for i in range(density.size):
		if ((index[i][0]>0 and index[i][0]<gridNumber) and (index[i][1]>0 and index[i][1]<gridNumber) and (index[i][2]>0 and index[i][2]<gridNumber)):
			field[index[i][0]][index[i][1]][index[i][2]] =field[index[i][0]][index[i][1]][index[i][2]] + density[i]
	return field

def getWavenumber(size,boxsize):
	axis = np.array([0. for i in range(size)])
	wavenumber = np.array(np.meshgrid(axis,axis,axis)[0])
	dk = 2*np.pi/boxsize
	Axis = np.arange(1,size+1,dtype =float)
	mesh = np.array(np.meshgrid(Axis,Axis,Axis))
	xindex = mesh[1]*mesh[1]
	yindex = mesh[0]*mesh[0]
	zindex = mesh[2]*mesh[2]
	wavenumber =dk*dk*(xindex+yindex+zindex)
	return wavenumber

def gaussianFilter(densityField,scale,wavenumber):
	GFilter = wavenumber*scale*scale/2
	GFilter = np.exp(-GFilter)
	print densityField[-1][-1][-1],GFilter[-1][-1][-1],wavenumber[-1][-1][-1]
	return densityField *GFilter

def computeDisplacement(densityField,wavenumber,boxsize):
	vectorAxis = np.array([[0.j,0.j,0.j] for i in range(densityField.shape[0])])
	displacementField = np.array([[vectorAxis for i in range(densityField.shape[0])]for i in range(densityField.shape[0])])
	dk = 2*np.pi/boxsize
	Axis = np.arange(1,densityField.shape[0]+1)
	mesh = np.array(np.meshgrid(Axis,Axis,Axis))
	xindex = mesh[1].transpose()
	yindex = mesh[0].transpose()
	zindex = mesh[2].transpose()
	displacementField = displacementField.transpose()
	displacementField[0] = densityField.transpose()*-1j*dk*xindex/wavenumber.transpose()
	displacementField[1] = densityField.transpose()*-1j*dk*yindex/wavenumber.transpose()
	displacementField[2] = densityField.transpose()*-1j*dk*zindex/wavenumber.transpose()
	displacementField = displacementField.transpose()
	return displacementField

def coordinateConvert(RA,DEC,z):
	Cart = np.array([[0. for i in range(RA.size)] for i in range(3)]) 
	Cart[0] = z*np.cos(np.radians(RA))*np.cos(np.radians(DEC)) 
	Cart[1] = z*np.sin(np.radians(RA))*np.cos(np.radians(DEC)) 
	Cart[2] = z*np.sin(np.radians(DEC)) 
	return Cart
	

def shiftParticle(original,displacement,gridNumber):
	boxRange = np.ogrid[original.min():original.max():gridNumber*1j]
	boxGrid = np.array([boxRange for i in range(3)])
	new = original.copy()
	xinterp = interpolate.interpn(boxGrid,np.fft.ifftn(displacement[0]),new).real
	yinterp = interpolate.interpn(boxGrid,np.fft.ifftn(displacement[1]),new).real
	zinterp = interpolate.interpn(boxGrid,np.fft.ifftn(displacement[2]),new).real
	new = new.transpose()
	new[0] = new[0] +xinterp
	new[1] = new[1] +yinterp
	new[2] = new[2] +zinterp
	new = new.transpose()
	return new



def generatingMask(densityField,binNumber,gridNumber,randomParticle = np.array([]),gridMax = 0.0):
	x = np.where(densityField!=0)[0]
	index = np.where(np.bincount(x)>0)[0]
	bins = np.array([0 for i in range(binNumber)])
	for i in range(binNumber):
		bins[i] = index[i*index.size/binNumber]
	ymax = np.array([0 for i in range(binNumber)])
	ymin = np.array([0 for i in range(binNumber)])
	zmax = np.array([0 for i in range(binNumber)])
	zmin = np.array([0 for i in range(binNumber)])
	for i in range(binNumber):
		while np.where(densityField[bins[i]]!=0)[0].size == 0:
			bins[i] = bins[i]+1
			print i,bins[i],np.where(densityField[bins[i]]!=0)[0].size
		ymax[i] = np.where(densityField[bins[i]]!=0)[0].max()
		ymin[i] = np.where(densityField[bins[i]]!=0)[0].min()
		zmax[i] = np.where(densityField[bins[i]]!=0)[1].max()
		zmin[i] = np.where(densityField[bins[i]]!=0)[1].min()
	maskRange = np.array([[0 for i in range(x.max()-x.min()+1)] for i in range(4)])
	xRange = np.ogrid[x.min():x.max():(x.max()-x.min()+1)*1j]
	maskRange[0] = np.interp(xRange,bins,ymax)
	maskRange[1] = np.interp(xRange,bins,ymin)
	maskRange[2] = np.interp(xRange,bins,zmax)
	maskRange[3] = np.interp(xRange,bins,zmin)
	if randomParticle.size == 0:
		print 'No random particle is supplied,generating mask array'
		axis = np.array([0. for i in range(gridNumber)])
		mask = np.array(np.meshgrid(axis,axis,axis)[0]) 
		for i in range(xRange.size):
			localaxis = axis.copy()
			localaxis[maskRange[3][i]:maskRange[2][i]] = 1
			mask[xRange[i]][maskRange[1][i]:maskRange[0][i]] = localaxis
		return mask
	else:
		print 'Masking random particle'
		randomParticle = randomParticle.T
		print randomParticle.shape
		randomIndex = np.floor(randomParticle*(gridNumber-1)/gridMax)	
		print randomIndex
		randomIndex[0] = np.where(np.logical_and((randomIndex[0]>x.min()),randomIndex[0]<x.max()),randomIndex[0],-1) 
		survivor = np.array([])
		for i in range(xRange.size):
			print np.where((randomIndex[0]==xRange[i]))[0].size,xRange[i],randomIndex[0]
			local = np.where((randomIndex[0]==xRange[i])*np.logical_and(randomIndex[1]<maskRange[0][i],randomIndex[1]>maskRange[1][i])*np.logical_and(randomIndex[2]<maskRange[2][i],randomIndex[2]>maskRange[3][i]))[0]
			print local.size
			survivor = np.append(survivor,local)
		survivor = np.array(survivor,dtype = int)
		randomParticle = randomParticle.T
		print survivor.size,randomParticle.size
		result = randomParticle[survivor]
		return result	

#def generatingMaskedField(field,mask):
	

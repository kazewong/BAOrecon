import numpy as np
from scipy import interpolate
from scipy.spatial import distance
def computeDensityField(position,density,gridNumber,boxsize,boxmin):
	axis = np.array([0. for i in range(gridNumber)])
	field = np.array(np.meshgrid(axis,axis,axis)[0])
	position = position.T
	index = position.copy()
	index[0] = np.floor((position[0]-boxmin)*(gridNumber-1)/boxsize)
	index[1] = np.floor((position[1]-boxmin)*(gridNumber-1)/boxsize)
	index[2] = np.floor((position[2]-boxmin)*(gridNumber-1)/boxsize)
	position = position.T
	index = index.T
	for i in range(density.size):
		if ((index[i][0]>=0 and index[i][0]<gridNumber) and (index[i][1]>=0 and index[i][1]<gridNumber) and (index[i][2]>=0 and index[i][2]<gridNumber)):
			field[index[i][0]][index[i][1]][index[i][2]] =field[index[i][0]][index[i][1]][index[i][2]] + density[i]
	return field

def gaussianFilter(densityField,scale,wavenumber):
	GFilter = wavenumber*scale*scale/2
	GFilter = np.exp(-GFilter)
	return densityField *GFilter

def getWavenumber(gridNumber,boxsize):
	axis = np.array([0. for i in range(gridNumber)])
	wavenumber = np.array(np.meshgrid(axis,axis,axis)[0])
	dk = 2*np.pi/boxsize
	Axis = np.arange(gridNumber,dtype =float)
	mesh = np.array(np.meshgrid(Axis,Axis,Axis))
	xindex = mesh[0]*mesh[0]
	yindex = mesh[1]*mesh[1]
	zindex = mesh[2]*mesh[2]
	wavenumber =dk*dk*(xindex+yindex+zindex)
	wavenumber[wavenumber==0]=1
	return wavenumber
	
def coordinateConvert(RA,DEC,z):
	Cart = np.array([[0. for i in range(RA.size)] for i in range(3)]) 
	Cart[0] = z*np.cos(np.radians(RA))*np.cos(np.radians(DEC)) 
	Cart[1] = z*np.sin(np.radians(RA))*np.cos(np.radians(DEC)) 
	Cart[2] = z*np.sin(np.radians(DEC)) 
	return Cart

def shiftParticle(original,displacement,gridNumber,boxsize,boxmin):
	boxRange = np.arange(gridNumber)
	boxGrid = np.array([boxRange for i in range(3)])
	new = original.copy()
	new[0] = (original[0]-boxmin)*(gridNumber-1)/boxsize
	new[1] = (original[1]-boxmin)*(gridNumber-1)/boxsize
	new[2] = (original[2]-boxmin)*(gridNumber-1)/boxsize
	xinterp = interpolate.interpn(boxGrid,displacement[0],new.T,bounds_error=False,fill_value = 0).real
	yinterp = interpolate.interpn(boxGrid,displacement[1],new.T,bounds_error=False,fill_value = 0).real
	zinterp = interpolate.interpn(boxGrid,displacement[2],new.T,bounds_error=False,fill_value = 0).real
	print 'The rms displacement is'
	print np.sqrt(np.sum(xinterp*xinterp+yinterp*yinterp+zinterp*zinterp)/(3*xinterp.size()))
	print 'The first 10 interpolated value of x displacement is'
	print new[0].min(),new[0].max(),new[1].min(),new[1].max(),new[2].min(),new[2].max(),xinterp.max(),yinterp.max(),zinterp.max(),xinterp.min(),yinterp.min(),zinterp.min()
	new[0] = (new[0])*(boxsize)/(gridNumber-1)+boxmin+xinterp
	new[1] = (new[1])*(boxsize)/(gridNumber-1)+boxmin+yinterp
	new[2] = (new[2])*(boxsize)/(gridNumber-1)+boxmin+zinterp
	new = new.transpose()
	return new

def maskDensityField(densityField,randomField):
	output = densityField.copy()
	mean = densityField[np.where(randomField!=0)].mean()
	#mask = generatingMask(randomField,10,densityField.shape[0])
	output[np.where(randomField==0)] = mean
	output = output/mean -1	
	return output
	
def getOffset(array,axis,distance):
	output = array.copy()
	if distance > 0:
		for i in range(distance):
			output = np.delete(output,0,axis)
		for i in range(distance):
			output = np.insert(output,output.shape[axis],0,axis)
	if distance < 0:
		for i in range(abs(distance)):
			output = np.delete(output,output.shape[axis]-1,axis)
		for i in range(abs(distance)):
			output = np.insert(output,0,0,axis)
	return output

def getCenterIndex(position,gridNumber,boxsize):
	position = position.T
	x = np.floor(0.-position[0].min()*(gridNumber-1)/boxsize)
	y = np.floor(0.-position[1].min()*(gridNumber-1)/boxsize)
	z = np.floor(0.-position[2].min()*(gridNumber-1)/boxsize)
	position = position.T
	return np.array([x,y,z])

def getDistance(gridNumber,boxsize,center):
	gridLength = boxsize/gridNumber
	Axis = np.arange(gridNumber)
	mesh = np.meshgrid(Axis,Axis,Axis)
	x = mesh[1]
	y = mesh[0]
	z = mesh[2]
	x = gridLength*(x-center[0])
	y = gridLength*(y-center[1])
	z = gridLength*(z-center[2])
	return np.array([x,y,z])
	
def getBoxSize(position):
	boxsize = 0.
	position = position.T
	for i in range(3):
		boxsize = boxsize + position[i].max()-position[i].min()
	position = position.T
	return boxsize/3

def getRadialFactor(gridNumber,boxsize,boxmin):
	axis=np.arange(float(gridNumber))
	mesh=np.array(np.meshgrid(axis,axis,axis))
	mesh-=np.floor(-boxmin*gridNumber/boxsize)
	output=mesh.copy()
	length=np.linalg.linalg.norm(mesh,axis=0)
	output[0]=mesh[0]/length
	output[1]=mesh[1]/length
	output[2]=mesh[2]/length
	output[np.where(np.isnan(output))]=0
	return output
	
	
def displacementSolver(dataField,randomField,gridSize):#The Jacobi method is used
	h= 1./gridSize
	h2 = 1./(gridSize**2)
	axis = np.array([0. for i in range(dataField.shape[0]+2)])
	potential = np.array(np.meshgrid(axis,axis,axis)[0])
	for i in range(100):
		potential[np.where(randomField==0)]=0
		for j in range(3):
			potential[1:gridSize+1,1:gridSize+1,1:gridSize+1] =  h2*dataField +getOffset(potential,j,1)[1:gridSize+1,1:gridSize+1,1:gridSize+1]+getOffset(potential,j,-1)[1:gridSize+1,1:gridSize+1,1:gridSize+1]
		potential = potential/6
	xdisplacement = (getOffset(potential[1:gridSize+1,1:gridSize+1,1:gridSize+1],0,1)-getOffset(potential[1:gridSize+1,1:gridSize+1,1:gridSize+1],0,-1))/(2*h)
	ydisplacement = (getOffset(potential[1:gridSize+1,1:gridSize+1,1:gridSize+1],1,1)-getOffset(potential[1:gridSize+1,1:gridSize+1,1:gridSize+1],1,-1))/(2*h)
	zdisplacement = (getOffset(potential[1:gridSize+1,1:gridSize+1,1:gridSize+1],2,1)-getOffset(potential[1:gridSize+1,1:gridSize+1,1:gridSize+1],2,-1))/(2*h)
	print "the min-max potential is:"+str(potential.min())+str(potential.max())
        print "The min-max x displacement is:"+str(xdisplacement.min())+str(xdisplacement.max())
        print "The min-max y displacement is:"+str(ydisplacement.min())+str(ydisplacement.max())
        print "The min-max z displacement is:"+str(zdisplacement.min())+str(zdisplacement.max())
	return np.array([xdisplacement,ydisplacement,zdisplacement])

####################trash can#####################
# The old function used ofr generating mask, using interpolation
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

def computeDisplacement(densityField,wavenumber,boxsize,bias):
	vectorAxis = np.array([[0.j,0.j,0.j] for i in range(densityField.shape[0])])
	displacementField = [0,0,0]
	dk = 2*np.pi/boxsize
	Axis = np.arange(densityField.shape[0])
	mesh = np.array(np.meshgrid(Axis,Axis,Axis))
	xindex = mesh[0]
	yindex = mesh[1]
	zindex = mesh[2]
	displacementField[0] = np.fft.ifftn(densityField*1j*dk*xindex/(bias*wavenumber))
	displacementField[1] = np.fft.ifftn(densityField*1j*dk*yindex/(bias*wavenumber))
	displacementField[2] = np.fft.ifftn(densityField*1j*dk*zindex/(bias*wavenumber))
	displacementField = np.array(displacementField)
	print 'The minmax disaplcement is'
	print displacementField.min(),displacementField.max()
	return np.array(displacementField)

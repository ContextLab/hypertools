##########NOTES#############
#reads numpy arrays


##########CLEANUP############
#ability to read other data formats?

#'trajectories must be specified in 2D matriices'
	#add this check!
	#line80 in matlab script

##########PACKAGES###########
import numpy as np

########MAIN FUNCTION########
def hyperplot(*args):

	"""
	Implements the "hyperalignment" algorithm described by the
	following paper:

	Haxby JV, Guntupalli JS, Connolly AC, Halchenko YO, Conroy BR, Gobbini
	MI, Hanke M, and Ramadge PJ (2011)  A common, high-dimensional model of
	the representational space in human ventral temporal cortex.  Neuron 72,
	404 -- 416.
	"""

	#use *args & args to allow multiple input arguments
	#creates a tuple, len==1


	###SECONDARY FUNCTIONS###
	def procrustes(X, Y, scaling=True, reflection='best'):
		"""
		This function copied from stackoverflow user ali_m 
		(http://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy) 
		8/22/16
	
		A port of MATLAB's `procrustes` function to Numpy.

		Procrustes analysis determines a linear transformation (translation,
		reflection, orthogonal rotation and scaling) of the points in Y to best
		conform them to the points in matrix X, using the sum of squared errors
		as the goodness of fit criterion.

		d, Z, [tform] = procrustes(X, Y)

		Inputs:
		------------
		X, Y    
			matrices of target and input coordinates. they must have equal
			numbers of  points (rows), but Y may have fewer dimensions
			(columns) than X.

		scaling 
			if False, the scaling component of the transformation is forced
			to 1

		reflection
			if 'best' (default), the transformation solution may or may not
			include a reflection component, depending on which fits the data
			best. setting reflection to True or False forces a solution with
			reflection or no reflection respectively.

		Outputs
		------------
		d       
			the residual sum of squared errors, normalized according to a
			measure of the scale of X, ((X - X.mean(0))**2).sum()

		Z
			the matrix of transformed Y-values

		tform   
			a dict specifying the rotation, translation and scaling that
			maps X --> Y

		"""

		n,m = X.shape
		ny,my = Y.shape

		muX = X.mean(0)
		muY = Y.mean(0)

		X0 = X - muX
		Y0 = Y - muY

		ssX = (X0**2.).sum()
		ssY = (Y0**2.).sum()

		# centred Frobenius norm
		normX = np.sqrt(ssX)
		normY = np.sqrt(ssY)

		# scale to equal (unit) norm
		X0 /= normX
		Y0 /= normY

		if my < m:
			Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

		# optimum rotation matrix of Y
		A = np.dot(X0.T, Y0)
		U,s,Vt = np.linalg.svd(A,full_matrices=False)
		V = Vt.T
		T = np.dot(V, U.T)

		if reflection is not 'best':

			# does the current solution use a reflection?
			have_reflection = np.linalg.det(T) < 0

			# if that's not what was specified, force another reflection
			if reflection != have_reflection:
				V[:,-1] *= -1
				s[-1] *= -1
				T = np.dot(V, U.T)

		traceTA = s.sum()

		if scaling:

			# optimum scaling of Y
			b = traceTA * normX / normY

			# standarised distance between X and b*Y*T + c
			d = 1 - traceTA**2

			# transformed coords
			Z = normX*traceTA*np.dot(Y0, T) + muX

		else:
			b = 1
			d = 1 + ssY/ssX - 2 * traceTA * normY / normX
			Z = normY*np.dot(Y0, T) + muX

		# transformation matrix
		if my < m:
			T = T[:my,:]
		c = muX - b*np.dot(muY, T)

		#transformation values 
		tform = {'rotation':T, 'scale':b, 'translation':c}

		return d, Z, tform

	def align(*args):

		sizes_0=[]

		#STEP 0: STANDARDIZE SIZE AND SHAPE	
		for x in range(0, len(args[0])):
			sizes_0[x]=args.shape[0][x]
			T=min(sizes_0)
			#find the smallest number of rows

			sizes_1=x.shape[1]
			sizes_1[x]=x.shape[1]
			T=max(sizes_1)
			#find the largest number of columns

		for x in args:
			x=x[0:T,:]
			#reduce each input argument to the minimum number of rows by deleting excess rows

			missing=T-x.shape[1]
			add=np.zeros((T, missing))
			y=np.append(x, add, axis=1)
			#add 'missing' number of columns (zeros) to each array

		#STEP 1: CREATE COMMON TEMPLATE
			#align first two subj's data, compute average of the two aligned data sets
		
			#for each subsequent subj:
			#align new subj to average of previous subjs; add this aligned subj data to the average	
		for x in range(0, len(args)):
			if x==0:
				template=args[x]
			else:
				next = procrustes((np.transpose(template/(x-1))), (np.transpose(args[x])))
				template = template + np.transpose(next)
		template= template/len(args)

		#STEP 2: NEW COMMON TEMPLATE
			#align each subj to the template from STEP 1
			#create a new template by the same means
		template2= numpy.zeros(template.shape)
		for x in range(0, len(args)):
			next = procrustes((np.transpose(template)),(np.transpose(args[x])))
			template2 = template2 + np.transpose(next)

		#STEP 3 (below): ALIGN TO NEW TEMPLATE
		for x in range(0, len(args)):
			next = procrustes((np.transpose(template2)),(np.transpose(args[x])))
			aligned[x] = np.transpose(next)


	#############BODY############
	if len(args)<=1:
		for x in range(0,len(args[0][:])):
			data_type[x]=type(args[0][x])

			if all(z==int for z in data_types):
				aligned=args[0]
				print "Only one dataset"

			elif all(z==np.ndarray for z in data_types):
				align(args[0][:])
				if each element of the array is a numpy array, then align elements to each other

			else: 
				print "Input argument elements are neither all ints nor all numpy arrays..."
				break

	else:
		if all(z==np.ndarray for z in args[0][:]):
			align(args[0][:])		
		else:
			print "Input datasets should be numpy arrays"
		#if each input argument is a numpy array, align them
				




		#align each input argument to the others

		
		#dims=[]
		#confused about how to deal with dimensions in python.. 
		#maybe len(x.shape) ??

		#need to check that trajectories are specified in 2d matrices







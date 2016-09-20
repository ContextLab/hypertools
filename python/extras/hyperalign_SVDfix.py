##########NOTES#############
#reads numpy arrays

##########CLEANUP############
#ability to read other data formats?

#'trajectories must be specified in 2D matriices'
	#add this check!
	#line80 in matlab script

##########PACKAGES###########
import numpy as np
import numpy as np,numpy.linalg

########MAIN FUNCTION########
def hyperalign(*args):

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
	def _getAplus(A):
		eigval, eigvec = np.linalg.eig(A)
		Q = np.matrix(eigvec)
		xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
		return Q*xdiag*Q.T

	def _getPs(A, W=None):
		W05 = np.matrix(W**.5)
		return  W05.I * _getAplus(W05 * A * W05) * W05.I

	def _getPu(A, W=None):
		Aret = np.array(A.copy())
		Aret[W > 0] = np.array(W)[W > 0]
		return np.matrix(Aret)

	def nearPD(A, nit=10):
		n = A.shape[0]
		W = np.identity(n) 
	# W is the matrix used for the norm (assumed to be Identity matrix here)
	# the algorithm should work for any diagonal W
		deltaS = 0
		Yk = A.copy()
		for k in range(nit):
			Rk = Yk - deltaS
			Xk = _getPs(Rk, W=W)
			deltaS = Xk - Rk
			Yk = _getPu(Xk, W=W)
		return Yk

	def is_pos_def(x):
		return np.all(np.linalg.eig(x)>0)

	def make_pos_def(x):
		if is_pos_def(x):
			return x
		else:
			return nearPD(x)

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
		#U,s,Vt = np.linalg.svd(A,full_matrices=False)
		U,s,Vt = np.linalg.svd(make_pos_def(A),full_matrices=False)
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

		#sizes_0=np.zeros((len(args)))
		#sizes_1=np.zeros((len(args)))

		#STEP 0: STANDARDIZE SIZE AND SHAPE	
		#for x in range(0, len(args)):
		#	sizes_0[x]=args[x].shape[0]
		#	T=min(sizes_0)
			#find the smallest number of rows

		#	sizes_1[x]=args[x].shape[1]
		#	T=max(sizes_1)
			#find the largest number of columns


		#for x in args:
		#	x=x[0:T,:]
			#reduce each input argument to the minimum number of rows by deleting excess rows

		#	missing=T-x.shape[1]
		#	add=np.zeros((T, missing))
		#	y=np.append(x, add, axis=1)
			#add 'missing' number of columns (zeros) to each array

			#TEST
		#	print args


		#STEP 1: CREATE COMMON TEMPLATE
			#align first two subj's data, compute average of the two aligned data sets
		
			#for each subsequent subj:
			#align new subj to average of previous subjs; add this aligned subj data to the average	
		for x in range(0, len(args)):
			if x==0:
				template=args[x]

				#TESTING
				print template
			
			else:
				next = procrustes(np.transpose(template/x), np.transpose(args[x]))
				template = template + np.transpose(next)
		template= template/len(args)

		

		#STEP 2: NEW COMMON TEMPLATE
			#align each subj to the template from STEP 1
			#create a new template by the same means
		template2= np.zeros(template.shape)
		for x in range(0, len(args)):
			next = procrustes((np.transpose(template)),(np.transpose(args[x])))
			template2 = template2 + np.transpose(next)

		#STEP 3 (below): ALIGN TO NEW TEMPLATE
		for x in range(0, len(args)):
			next = procrustes((np.transpose(template2)),(np.transpose(args[x])))
			aligned[x] = np.transpose(next)


		#TESTING
		print aligned



#############MAIN FUNCTION BODY############
	if len(args)<=1:

		if all(isinstance(x, int) for x in args[0]):
			aligned=args
			print "Only one dataset"
			return aligned

		elif all(isinstance(x, np.ndarray) for x in args[0]):
			align(*args)
			#if each element of the input is a numpy array, then align elements to each other
			print "single array"

		else: 
			print "Input argument elements are neither all ints nor all numpy arrays..."

	else:
		if all(isinstance(x, np.ndarray) for x in args):
			align(*args)	
			print "multiple arrays"

		else:
			print "Input datasets should be numpy arrays"
		#if each input argument is a numpy array, align them
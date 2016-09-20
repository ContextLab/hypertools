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
def hyperalign(*args):

	"""
	Implements the "hyperalignment" algorithm described by the
	following paper:

	Haxby JV, Guntupalli JS, Connolly AC, Halchenko YO, Conroy BR, Gobbini
	MI, Hanke M, and Ramadge PJ (2011)  A common, high-dimensional model of
	the representational space in human ventral temporal cortex.  Neuron 72,
	404 -- 416.
	"""

	###FUNCTIONS###
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

		return Z

	def align(*args):

		sizes_0=np.zeros((len(args)))
		sizes_1=np.zeros((len(args)))

		print sizes_0 #[0,0]
		print sizes_1#[0,0]

		#STEP 0: STANDARDIZE SIZE AND SHAPE	
		for x in range(0, len(args)):

			sizes_0[x]=args[x].shape[0]
			sizes_1[x]=args[x].shape[1]
			
		print sizes_0#[3,2]
		print sizes_1#[3,6]

		R=min(sizes_0)
		#find the smallest number of rows
		C=max(sizes_1)
		#find the largest number of columns

		print R
		print C

		k=np.empty((R,C), dtype=np.ndarray)
		m=[k]*len(args)


		for x in args:
			for y in range(0,len(args)):
				print x
#
				x=x[0:R,:]
				#reduce each input argument to the minimum number of rows by deleting excess rows

				print x

				missing=C-x.shape[1]
				
				print missing

				add=np.zeros((x.shape[0], missing))
				
				print add
				x=np.append(x, add, axis=1)
				m[y]=x



		for x in range(0, len(m)):
			if x==0:
				template=m[x]
			
			else:
				next = procrustes(np.transpose(template/x), np.transpose(m[x]))
				template = template + np.transpose(next)
		
		template= template/len(m)


		#STEP 2: NEW COMMON TEMPLATE
			#create a new template by the same means

		template2= np.zeros(template.shape)
		for x in range(0, len(m)):
			next = procrustes(np.transpose(template),np.transpose(m[x]))
			template2 = template2 + np.transpose(next)

		template2=template2/len(m)

			#align each subj to the template from STEP 1

		#STEP 3: ALIGN TO NEW TEMPLATE

		empty= np.zeros(template2.shape)
		aligned=[empty]*(len(m)) 

		for x in range(0, len(m)):
			next = procrustes(np.transpose(template2),np.transpose(m[x]))
			aligned[x] = np.transpose(next)

		return aligned
		print aligned


	##PARSE INPUT##
	if len(args)<=1:
		if all(isinstance(x, int) for x in args[0]):
			aligned=args
			print "Only one dataset"
			return aligned

		elif all(isinstance(x, np.ndarray) for x in args[0]):
			print "single array"
			return align(*args)
			#if each element of single input is a numpy array, align elements to each other

		else: 
			print "Input argument elements are neither all ints nor all numpy arrays..."

	else:
		if all(isinstance(x, np.ndarray) for x in args):
			print "multiple arrays"
			return align(*args)	
			#if each input is an array, align inputs to each other

		else:
			print "Input datasets should be numpy arrays"

			
import numpy as np
def resize(*args):

	sizes_0=np.zeros((len(args)))
	sizes_1=np.zeros((len(args)))

	#STEP 0: STANDARDIZE SIZE AND SHAPE	
	for x in range(0, len(args)):

		sizes_0[x]=args[x].shape[0]
		sizes_1[x]=args[x].shape[1]

	R=min(sizes_0)
	#find the smallest number of rows
	C=max(sizes_1)
	#find the largest number of columns

	k=np.empty((R,C), dtype=np.ndarray)
	m=[k]*len(args)
	
	for idx,x in enumerate(args):
		y=x[0:R,:]
		#reduce each input argument to the minimum number of rows by deleting excess rows
		
		missing=C-y.shape[1]
		add=np.zeros((y.shape[0], missing))
		y=np.append(y, add, axis=1)

		m[idx]=y
			#add 'missing' number of columns (zeros) to each array

			#TEST
	print m




#resizing with less loops
#vectorizing OR apply_along_axis OR functions
#prob functions
#
#one function to return rows and columns
#given np arrray size (a,b), return np array of (R,C), where the rows might be truncated and cols expanded
#---> may want two functions inside to truncate rows and add cols

#copy 
#--> pass by ref versus passing by value
#operating on object x or what x refers to
#ehat I want is what x referes to, not the value of x




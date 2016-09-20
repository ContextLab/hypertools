import numpy as np
def test(*args):	




	##THIS CODE PARSES THE INPUT DATA CORRECTLY

	if len(args)<=1:
		if all(isinstance(x, int) for x in args[0]):
			aligned=args
			print "Only one dataset"
			return aligned

		elif all(isinstance(x, np.ndarray) for x in args[0]):
			print "single array"
			#align(*args)
			#if each element of the input is a numpy array, then align elements to each other

		else: 
			print "Input argument elements are neither all ints nor all numpy arrays..."

	else:
		if all(isinstance(x, np.ndarray) for x in args):
			print "multiple arrays"
			#align(*args)	
			

		else:
			print "Input datasets should be numpy arrays"
		#if each input argument is a numpy array, align them
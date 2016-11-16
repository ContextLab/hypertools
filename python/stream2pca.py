from matplotlib import style
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import scipy
from scipy.interpolate import PchipInterpolator as pchip
#import PCA as PCA
from mpl_toolkits.mplot3d import Axes3D
import hyperalign as hyp
from sklearn.decomposition import PCA
#import threadin
import trollius
from trollius import From
import csv
import os


def stream(raw_file, pca_file, seconds, rate):

	##VARIABLES##

	required_lines=8
	#required_lines=seconds*rate
	sleep_secs=1  
	#sleep_secs=1/rate 
	csv1 = np.genfromtxt(raw_file, delimiter=",")
	csv2 = np.genfromtxt(pca_file, delimiter=",")
	head=6 #number of header rows
	chan=1#number of leading channel-info cols
	skip=head+required_lines
	session_secs=6000 

####################################

	@trollius.coroutine
	def count_lines():
		num_lines=open(raw_file).read().count('\n')
		while num_lines != required_lines:
			yield From(trollius.sleep(sleep_secs))
			num_lines=open(raw_file).read().count('\n')
			
	loop = trollius.get_event_loop()
	loop.run_until_complete(count_lines())
	#loop.close()
	#loop until minimum line thresh reached, then PCA



	@trollius.coroutine
	def PCA_append():
		num_lines2=open(pca_file).read().count('\n')
		while num_lines2 < session_secs*rate:
			x1=csv1[:, 1:] #ignore header and channel info
			z=x1[0,:]
			m=PCA(n_components=3, whiten=True)
			m.fit(x1)
			#m.fit(x[0])
			#PCA transform (to first row)


			if os.stat(pca_file).st_size ==0:
				w = csv.writer(open(pca_file,'a'),dialect='excel')
				first=x1[-1,:]
				w.writerows(first)

			else:
				y=csv2[-1,:]
				p=m.transform(z)
				if not np.array_equal(p, y):
					#if statement to protect against repeats/timing errors
					w = csv.writer(open(pca_file,'a'),dialect='excel')
					w.writerows(p)

		loop = trollius.get_event_loop()
		loop.run_until_complete(PCA_append())

		















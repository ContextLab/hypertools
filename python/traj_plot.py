#!/usr/bin/env python

"""

"""

##PACKAGES##
import numpy as np

##META##
__authors__ = ["Jeremy Manning", "Kirsten Ziman"]
__version__ = "1.0.0"
__maintainers__ = ["Jeremy Manning", "Kirsten Ziman"] 
__emails__ = ["Jeremy.R.Manning@dartmouth.edu", "kirstenkmbziman@gmail.com", "contextualdynamics@gmail.com"]
#__copyright__ = ""
#__credits__ = [""]
#__license__ = ""

#FIRST STEPS
#++++++++++++++++
#-PCA (3D)
#	-essentially same base as plot_coords, but different animation.. 
#-smooth to 100 samples per window (p chip to scale up by ten)
#

#THOUGHTS
#+++++++++++++++++
#why do PCA and plotting again when plot_coords already does it?
#better to rewrite this as standalone script or to let it call on other scripts in the package?
from matplotlib import style
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import scipy
from scipy.interpolate import PchipInterpolator as pchip
from mpl_toolkits.mplot3d import Axes3D
import hyperalign as hyp
from sklearn.decomposition import PCA
import trollius
from trollius import From
import csv
import os
import time


def stream(raw_file, pca_file, seconds, rate):

    ##VARIABLES##
    
    required_lines=seconds*rate #required lines before computing PCA
    sleep_secs=1/rate #seconds to sleep before checking for new data
    csv1 = np.genfromtxt(raw_file, delimiter=",")
    csv2 = np.genfromtxt(pca_file, delimiter=",")
    head=0 #number of header rows
    chan=0 #number of leading channel-info cols

    #session_secs=600 
    #set this parameter and change commenting in last while loop for predefined session length


####################################


    num_lines=open(raw_file).read().count('\n')

    while num_lines==0:
        time.sleep(sleep_secs)
        num_lines=open(raw_file).read().count('\n')
        #if no data, rest

    while num_lines != required_lines:
        time.sleep(sleep_secs)
        num_lines=open(raw_file).read().count('\n')
        #while there is data, but not enough for PCA, rest



    x1=csv1[head:,chan:] #x1 is raw data minus header and channel info        
    m=PCA(n_components=3, whiten=True)
    m.fit(x1)
    #compute PCA


    out= m.transform(x1[-1,:])
    w = csv.writer(open(pca_file,'a'),dialect='excel')
    w.writerows(out)
    #write out the last PCA interval datapoint as first plot point


    num_lines2=open(pca_file).read().count('\n')
    #while num_lines2 < session_secs*rate :
    while True:
        csv_stream = np.genfromtxt(raw_file, delimiter=",")
        csv_pca = np.genfromtxt(pca_file, delimiter=",")


        p=m.transform(csv_stream[-1,:])            
        w = csv.writer(open(pca_file,'a'),dialect='excel')
        w.writerows(p)
        time.sleep(sleep_secs)
    #num_lines2=open(pca_file).read().count('\n')

    
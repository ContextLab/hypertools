import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import time

style.use('fivethirtyeight')

fig=plt.figure()
ax1=fig.add_subplot(1, 1, 1)

def animate(i):
	graph_data = open('samplefile.txt','r').read()
	lines=graph_data.split('\n')

	xs=[]
	ys=[]

	for line in lines:
		if len(line)>1:
			x,y=line.split(',')
			xs.append(x)
			ys.append(y)
	ax1.clear
	ax1.plot(xs, ys)
ani=animation.FuncAnimation(fig, animate, interval=2000)

#plt.show()

#data=np.array([[3, 7],[1, 2],[8, 11],[5, -12],[20, 25], [-3, 30], [2,2], [17, 17])

def animate2(i):
	fig=plt.figure()
	ax1=fig.add_subplot(1, 1, 1)

	k=1

	while k <=len(i):
		ax1.clear
		ax1.plot(i[0:k-1, 0], i[0:k-1, 1])
		#make sure to include previous points, append
		
		plt.show()
		time.sleep(1)
		k=k+1


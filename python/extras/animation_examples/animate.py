import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
def animate2(i):
	ax1.clear()
	ax1.plot(test_data[0:i, 0], test_data[0:i, 1])
test_data=np.array([[3, 7],[1, 2],[8, 11],[5, -12],[20, 25], [-3, 30], [2,2], [17, 17]])
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
animation.FuncAnimation(fig, animate2, frames=range(1, len(test_data)), interval=500, repeat=True)
import pickle
import time
import sprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print('This program will animate the time dependent density from a ProbPsi.db file')
Nx = int(input('(In ProbPsi file name) Nx = '))
Nt = int(input('(In ProbPsi file name) Nt = '))
x0 = float(input('x_min = '))
x1 = float(input('x_max = '))
dx = float(input('deltax = '))
input_file = open('ProbPsi(Nx=' + str(Nx) + ',Nt=' + str(Nt) + ').db','r')
data = pickle.load(input_file)
plt.ion()
xx = np.arange(-x1,(x1 + (dx/2.0)), dx)  
line, = plt.plot(xx,data[0])
plt.ylim(ymax=1.0) 
plt.ylim(ymin=-0.1) 
plt.xlim(xmax=x1) 
plt.xlim(xmin=x0)
i = 0
while(i < len(data)):
	line.set_ydata(data[i][:])
	plt.draw()    
	string = 'Animating Density: i = ' + str(i)
	sprint.sprint(string,1,1,1)
	time.sleep(0.01)
	i += 1
input_file.close()
print

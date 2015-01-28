import os
import pickle
import time
import sprint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print('This program will animate the time dependent density from a ProbPsi.db file and produce an .avi movie of the animation')
Nx = int(input('(In ProbPsi file name) Nx = '))
Nt = int(input('(In ProbPsi file name) Nt = '))
x0 = float(input('x_min = '))
x1 = float(input('x_max = '))
dx = float(input('deltax = '))
f = str(raw_input('output animation file name = '))
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
	fname = '_tmp%03d.png'%i        
	matplotlib.pyplot.savefig(fname) 
	string = 'animating density: i = ' + str(i) + '/' + str(len(data))
	sprint.sprint(string,1,1,1)
	time.sleep(0.01)
	i += 1
print
print('making movie animation (avi): this make take a while')
os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o " + str(f))
j=0
while (j <= i-1):
	os.remove('_tmp%03d.png'%(j))
	j += 1
input_file.close()


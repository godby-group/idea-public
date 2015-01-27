#!/usr/bin/python

import parameters as pm
import os
import sys
import pickle
import numpy as np
import time
import sprint
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TD = pm.TD
jmax = pm.jmax
kmax = pm.kmax
msglvl = pm.msglvl

if int(TD) == 1:
	tmax = pm.tmax
	imax = pm.imax
	deltat = tmax/(imax-1)
if int(TD) == 0:
	tmax = 0.0
	imax = 1
	deltat = 0.0

xmax = pm.xmax
deltax = pm.deltax
Buffer = pm.Buffer

ProbDensity = np.zeros((imax,jmax), dtype = np.float)

def ProbabilityDensity():
    i = 0
    origdir = os.getcwd()    
    newdir = os.path.join('MPsiReal_binaries')
    if not os.path.isdir(newdir):
        print 'error: files do not exist. Run ED_run_bicg.py'
    Particle = open('Particle2P.txt', 'w')
    dodah=open('max','w')
    check=open('monitor', 'w')
    check1=open('monitorright','w')
    check2=open('monitormin', 'w')
    initial=open('initial','w')
    final=open('final','w')
    
    while (i < imax):
        j = 0
        os.chdir(newdir)        
        Psi2D = np.load('MPsi2DReal_%i.npy' %(i))
        os.chdir(origdir)
        ProbDensity[i,:] = np.sum(Psi2D[:,:], axis=0)*deltax*2.0
        a=np.amax(ProbDensity[i,:])
        dodah.write("%g\n" %a)
        text = '\n Time Step =' + str(i) + ', Integral of modulus over all space, =' +  str(np.sum(np.absolute(ProbDensity[i,:]))*(deltax))
        Particle.write(text)
        string = 'Calculating Density: t = ' + str(i) + ', Normalisation = ' + str(np.sum(ProbDensity[i,:])*deltax)
	sprint.sprint(string,1,1,msglvl)
	string = 'Calculating Density: t = ' + str(i) + ', Normalisation = ' + str(np.sum(ProbDensity[i,:])*deltax)
	sprint.sprint(string,2,1,msglvl)
        if (i==1):
            while (j<jmax-1):
                x = -xmax*1.0 + (j*deltax/1.0)
                initial.write("%5.10g\t%g\n" %(x,ProbDensity[i,j]))
                j+=1
	if (i==imax-1):
	    while (j<jmax-1):
		x = -xmax+(j*deltax/1.0)
		final.write("%5.10g\t%g\n" %(x,ProbDensity[i,j]))
		j+=1
        j=2
        peak=1
	jj=1
        while (j<jmax-2):
            if (ProbDensity[i,j+1]<ProbDensity[i,j] and ProbDensity[i,j-1]<ProbDensity[i,j] and ProbDensity[i,j]-ProbDensity[i,j+1]<0.001):
                if (ProbDensity[i,j]>0.1):
                    x = -xmax*1.0 + (j*deltax/1.0)                                       
                    if (peak==1) and (jj !=0):     
                        check.write('%5.8f\t%5g\n' %(ProbDensity[i,j],x))
                        #print peak,j
                        peak+=1
			jj=0
			jjj=j
                    if (peak==2) and (jjj !=j):   
                        check1.write('%5.8f\t%5g\n' %(ProbDensity[i,j],x))
                        #print peak,j
                        peak+=1
            if (ProbDensity[i,j+2]>ProbDensity[i,j] and ProbDensity[i,j-2]>ProbDensity[i,j] and ProbDensity[i,j]-ProbDensity[i,j+1]<0.001):
                x = -xmax*1.0 + (j*deltax/1.0)
                if (ProbDensity[i,j]>0.01):
                    check2.write('%5.8f\t%5g\n' %(ProbDensity[i,j],x))
            j+=1
        j=0
        i = i + 1	
    print
    check.close()
    return ProbDensity

#def Plotting(ProbPsi):
#        Writer = animation.writers['ffmpeg']
#        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#	plt.ion()
#	tstart = time.time()  
#	xx = np.arange(-xmax,(xmax + (deltax/2.0)), deltax)  
#	
#	line, = plt.plot(xx,ProbPsi[1,:])
#
#	plt.ylim(ymax=0.5) 
#	plt.ylim(ymin=0.0) 
#	plt.xlim(xmax=xmax) 
#	plt.xlim(xmin=-xmax)
#	count=0
#	i = 0
#	while (i < imax-1):
#		line.set_ydata(ProbPsi[i,:])
#		plt.draw()  
#		fname = '_tmp%03d.png'%i        
#		matplotlib.pyplot.savefig(fname)   
#		string = 'Plotting Density: t = ' + str(i)
#		sprint.sprint(string,1,1,msglvl)
#		string = 'Plotting Density: t = ' + str(i)
#		sprint.sprint(string,2,1,msglvl)
#		count+=1
#		time.sleep(0.1)
#		i = i + 1
#	print
#	string = 'Making movie animation.mpg: this make take a while'
#	sprint.sprint(string,1,0,msglvl)
#       string = 'Making movie animation.mpg: this make take a while'
#	sprint.sprint(string,2,0,msglvl)
#	print 
#	os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o animation_Nt=%s_tmax=%s.avi" %(imax,tmax))
#	j=0
#	while (j <= count-1):
#	    os.remove('_tmp%03d.png'%(j))
#	    j+=1
#	return
	
def Run():
    ProbDensity = ProbabilityDensity()
    ProbPsiFile = open("ProbPsi(Nx=%s,Nt=%s).db" % (jmax,imax),"w")
    pickle.dump(ProbDensity, ProbPsiFile)
    ProbPsiFile.close()
    #if int(TD) == 1:
    #	Plotting(ProbDensity)
    #f = open('Ground-stateDensity.out', 'w')
    #for j in range(jmax):
    #   f.write('%s %s\n' % (j*deltax-xmax, ProbDensity[0,j].real))
    #f.close()

    return

if(__name__ == '__main__'):
    Run()

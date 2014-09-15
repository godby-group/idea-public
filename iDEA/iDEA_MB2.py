# Library Imports
import os
import mkl
import time
import math
import copy
import pickle
import sprint
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import special
from scipy import sparse
from scipy.sparse import linalg as spla
import parameters as pm
from antisym2e import *

# Varaibale initialisation
jmax = pm.jmax
kmax = pm.kmax
xmax = pm.xmax
tmax = pm.ctmax
deltax = pm.deltax
deltat = pm.deltat
imax = int((tmax/deltat))+1
ctol = pm.ctol
rtol = pm.rtol
TD = pm.TD
par = pm.par
msglvl = pm.msglvl
c_m = 0
c_p = 0 
Nx_RM = 0

# Takes every combination of the two electron indicies and creates a single unique index 
def Gind(j,k):
    return (k + j*jmax)
    
# Inverses the Gind operation. Takes the single index and returns the corresponding indices used to create it.
def InvGind(jk):
    k = jk % jmax
    j = (jk - k)/jmax
    return j, k

# Calculates the nth Energy Eigenfunction of the Harmonic Oscillator (~H(n)(x)exp(x^2/2))
def EnergyEigenfunction(n):
    j = 0
    x = -xmax
    Psi = np.zeros(jmax, dtype = np.cfloat)
    while (x < xmax):
        factorial = np.arange(0, n+1, 1)
        fact = np.product(factorial[1:])
        norm = (np.sqrt(1.0/((2.0**n)*fact)))*((1.0/math.pi)**0.25)
        Psi[j] = complex(norm*(sp.special.hermite(n)(x))*(0.25)*np.exp(-0.5*(0.25)*(x**2)), 0.0)  
        j = j + 1
        x = x + deltax
    return Psi

# Builds the Cayley Matrix on the LHS (C1) of the matrix eigenproblem in Complex Time (t*=-it): C1=(I+H.dt/2)
def Matrixdef(i,r):
    j = 0
    k = 0
    while (j < jmax):
        k = 0
        while (k < kmax):
            jk = Gind(j,k)
            a=0;b=0;c=0;d=0
            Mat[jk,jk] = 1.0 + (4.0*r) + (2.0*r*(deltax**2)*(pm.Potential(i,j,k)))
            aa=Mat[jk,jk]          
            if (j < jmax - 1):
                Mat[jk,Gind(j+1,k)] = - r
                a=Mat[jk,Gind(j+1,k)]
            if (j > 0):
                Mat[jk,Gind(j-1,k)] = - r
                b=Mat[jk,Gind(j-1,k)]
            if (k < kmax - 1):
                Mat[jk,Gind(j,k+1)] = - r
                c=Mat[jk,Gind(j,k+1)]
            if (k > 0):
                Mat[jk,Gind(j,k-1)] = - r
                d=Mat[jk,Gind(j,k-1)]
            k = k + 1
        j = j + 1
    return Mat

# Imaginary Time Crank Nicholson initial condition
def InitialconI():
    Psi1 = np.zeros(jmax,dtype = np.cfloat)
    Psi2 = np.zeros(kmax,dtype = np.cfloat)
    Psi1 = EnergyEigenfunction(0)
    Psi2 = EnergyEigenfunction(1)
    j = 0
    while (j < jmax):
        k = 0
        while (k < kmax):
            Pair = Psi1[j]*Psi2[k] - Psi1[k]*Psi2[j]
            Psiarr[0,Gind(j,k)] = Pair
            k = k + 1
        j = j + 1
    return Psiarr[0,:]

# Takes the ground state solution and uses it as the initial eigenfunction in the real propagation
def InitialconR():
    Psi2D = np.zeros((jmax,kmax), dtype = np.cfloat)
    Psi2D = np.load('Psi_IC_Real2.npy')
    Psiarr[0,:] = PsiInverter(Psi2D[:,:],0)
    return Psiarr[0,:]

# Define function to turn array of compressed indexes into seperated indexes
def PsiConverterI(Psiarr,i):
    Psi2D = np.zeros((jmax,kmax), dtype = np.cfloat)
    mPsi2D = np.zeros((jmax,kmax))
    jk = 0
    while (jk < jmax**2):
        j, k = InvGind(jk)
        Psi2D[j,k] = Psiarr[jk]
        jk = jk + 1
    mPsi2D[:,:] = (np.absolute(Psi2D[:,:])**2)
    np.save('Psi_IC_Real', Psiarr[:])
    np.save('Psi_IC_Real2', Psi2D[:,:])
    return

# Define function to turn array of compressed indexes into seperated indexes
def PsiConverterR(Psiarr,i):
    Psi2D = np.zeros((jmax,kmax), dtype = np.cfloat)
    mPsi2D = np.zeros((jmax,kmax))
    jk = 0
    while (jk < jmax**2):
        j, k = InvGind(jk)
        Psi2D[j,k] = Psiarr[jk]
        jk = jk + 1
    mPsi2D[:,:] = (np.absolute(Psi2D[:,:])**2)
    origdir = os.getcwd()
    newdir = os.path.join('MPsiReal_binaries')
    if not os.path.isdir(newdir):
        os.mkdir(newdir)
    os.chdir(newdir)
    np.save('MPsi2DReal_%i' %(i),mPsi2D)
    np.save('Psi2DReal_%i' %(i),Psi2D)
    os.chdir(origdir)
    return 

# Psi inverter
def PsiInverter(Psi2D,i):
    Psiarr = np.zeros((jmax**2), dtype = np.cfloat)
    j = 0
    k = 0
    while (j < jmax):
        k = 0
        while (k < kmax):
            jk = Gind(j,k)
            Psiarr[jk] = Psi2D[j,k]
            k = k + 1
        j = j + 1
    return Psiarr[:]
	
# Plotting Function
def Plotter():
    origdir = os.getcwd()
    newdir = os.path.join('MPsiReal_binaries')
    Psi = np.load('Psi_IC_Real2.npy')
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-xmax,(xmax+(deltax/2.0)), deltax)
    Y = np.arange(-xmax,(xmax+(deltax/2.0)), deltax)
    X, Y = np.meshgrid(X, Y)
    ax.plot_wireframe(X, Y, Psi[:,:], rstride = 1, cstride = 1, linewidth=0.1, antialiased=True)
    plt.show()
    return

# Function to calulate energy of a wavefuntion
def Energy(Psi):
    a = np.linalg.norm(Psi[0,:])
    b = np.linalg.norm(Psi[1,:])
    return -(np.log(b/a))/deltat

# Function to construct the real matrix Af 
def ConstructAf(A):
    A1_dat, A2_dat = mkl.mkl_split(A.data,len(A.data))
    A.data = A1_dat
    A1 = copy.copy(A)
    A.data = A2_dat
    A2 = copy.copy(A)
    Af = sp.sparse.bmat([[A1,-A2],[A2,A1]]).tocsr()
    return Af

# Function to iterate over complex time
def CNsolveComplexTime():
    i = 1

    # Set the initial condition of the wavefunction
    Psiarr[0,:] = InitialconI()
    Psiarr_RM = c_m*Psiarr[0,:]

    # Construct the matrix A
    A = Matrixdef(0,r)
    A = A.tocsc()
    A_RM = c_m*A*c_p

    # Construct the matrix C
    C = -(A-sp.sparse.identity(jmax**2, dtype=np.cfloat))+sp.sparse.identity(jmax**2, dtype=np.cfloat)
    C_RM = c_m*C*c_p

    # Perform iterations
    while (i < imax):

	# Begin timing the iteration
        start = time.time()
        string = 'Complex Time = ' + str(i*deltat)
	sprint.sprint(string,2,0,msglvl)
        
	# Reduce the wavefunction
        if (i>=2):
            Psiarr[0,:]=Psiarr[1,:]
            Psiarr_RM = c_m*Psiarr[0,:]

	# Construct vector b
	if(par == 0):
	    b_RM = C_RM*Psiarr_RM
	else:
	    b_RM = mkl.mkl_mvmultiply_c(C_RM.data,C_RM.indptr+1,C_RM.indices+1,1,Psiarr_RM,C_RM.shape[0],C_RM.indices.size)

	# Solve Ax=b
	Psiarr_RM,info = spla.cg(A_RM,b_RM,x0=Psiarr_RM,tol=ctol)

	# Expand the wavefunction
        Psiarr[1,:] = c_p*Psiarr_RM

	# Calculate the energy
        Ev = Energy(Psiarr)
	string = 'Energy = ' + str(Ev)
	sprint.sprint(string,2,0,msglvl)

	# Normalise the wavefunction
	mag = (np.linalg.norm(Psiarr[1,:])*deltax)
        Psiarr[1,:] = Psiarr[1,:]/mag
	
	# Stop timing the iteration
	finish = time.time()
        string = 'Time to Complete Step: ' + str(finish-start)
	sprint.sprint(string,2,0,msglvl)

	# Test for convergance
	wf_con = np.linalg.norm(Psiarr[0,:]-Psiarr[1,:])
	string = 'Wave Function Convergence: ' + str(wf_con)
	sprint.sprint(string,2,0,msglvl)
	string = 'Many Body Complex Time: ' + 't = ' + str(i*deltat) + ', Convergence = ' + str(wf_con)
        sprint.sprint(string,1,1,msglvl)
	if(i>1):
	    e_con = old_energy - Ev
	    string = 'Energy Convergence: ' + str(e_con)
	    sprint.sprint(string,2,0,msglvl)
	    if(e_con < ctol*10.0 and wf_con < ctol*10.0):
		print
	        string = 'Many Body Complex Time: Ground State Converged' 
		sprint.sprint(string,1,0,msglvl)
                string = 'Ground State Converged' 
		sprint.sprint(string,2,0,msglvl)
	        i = imax
	old_energy = copy.copy(Ev)
        string = '---------------------------------------------------'
	sprint.sprint(string,2,0,msglvl)

        # Iterate
        i += 1

    # Convert Psi
    PsiConverterI(Psiarr[1,:],i)

    # Dispose of matrices and terminate
    A = 0
    C = 0
    return 

# Function to iterate over real time
def CNsolveRealTime():
    i = 1

    # Set the initial condition of the wavefunction
    Psiarr[0,:] = InitialconR()
    PsiConverterR(Psiarr[0,:],0)
    Psiarr_RM = c_m*Psiarr[0,:]
    Particle = open('Particle.txt', 'w')

    # Construct the matrix A
    A = Matrixdef(i,r)
    A = A.tocsc()
    A_RM = c_m*A*c_p

    # Construct the matrix Af if neccessary
    if(par == 1):
        Af = ConstructAf(A_RM)

    # Construct the matrix C
    C = -(A-sp.sparse.identity(jmax**2, dtype=np.cfloat))+sp.sparse.identity(jmax**2, dtype=np.cfloat)
    C_RM = c_m*C*c_p
    
    # Perform iterations
    while (i < imax):

	# Begin timing the iteration
        start = time.time()
	string = 'Real Time = ' + str(i*deltat) + '/' + str((imax-1)*deltat)
	sprint.sprint(string,2,0,msglvl)

	# Reduce the wavefunction
        if (i>=2):
            Psiarr[0,:] = Psiarr[1,:]

        # Construct the vector b
	b = C*Psiarr[0,:]
        if(par == 0):
	    b_RM = C_RM*Psiarr_RM
	else:
	    b_RM = mkl.mkl_mvmultiply_c(C_RM.data,C_RM.indptr+1,C_RM.indices+1,1,Psiarr_RM,C_RM.shape[0],C_RM.indices.size)

	# Solve Ax=b
	if(par == 0):
	    Psiarr_RM,info = spla.cg(A_RM,b_RM,x0=Psiarr_RM,tol=rtol)
	else:
	    b1, b2 = mkl.mkl_split(b_RM,len(b_RM))
	    bf = np.append(b1,b2)
	    if(i == 1):
		xf = bf
	    xf = mkl.mkl_isolve(Af.data,Af.indptr+1,Af.indices+1,1,bf,xf,Af.shape[0],Af.indices.size)
	    x1, x2 = np.split(xf,2)
	    Psiarr_RM = mkl.mkl_comb(x1,x2,len(x1))

        # Expand the wavefunction
	Psiarr[1,:] = c_p*Psiarr_RM

	# Convert the wavefunction
        PsiConverterR(Psiarr[1,:],i)
   
	# Write to file
        text = '\n Time Step =' + str(i) + ', Integral of modulus over all space, ie. P(particle1 and particle2)='
	text = text + str(np.sum(np.absolute(Psiarr[1,:])**2)*(deltax**2))
        Particle.write(text)

	# Stop timing the iteration
	finish = time.time()
        string = 'Time to Complete Step: ' + str(finish-start)
	sprint.sprint(string,2,0,msglvl)

	# Print to screen
        string = 'Residual: ' + str(np.linalg.norm(A*Psiarr[1,:]-b))
	sprint.sprint(string,2,0,msglvl)
	normal = np.sum(np.absolute(Psiarr[1,:])**2)*(deltax**2)
        string = 'Normalisation: ' + str(normal)
	sprint.sprint(string,2,0,msglvl)
	string = 'Many Body Real Time: ' + 't = ' + str(i*deltat) + ', Normalisation = ' + str(normal)
        sprint.sprint(string,1,1,msglvl)
        string = '---------------------------------------------------'
	sprint.sprint(string,2,0,msglvl)

	# Iterate
        i += 1

    # Dispose of matrices and terminate
    A = 0
    C = 0
    Particle.close()
    sprint.sprint(' ',1,0,msglvl)
    return

# Call this function to run iDEA-MB for 2 electrons
def main():

    # Use global variables
    global jmax,kmax,xmax,tmax,deltax,deltat,imax,msglvl,Psiarr,Mat,Rhv2,Psi2D,r,Mat2,c_m,c_p,Nx_RM

    # Construct reduction and expansion matrices
    c_m, c_p, Nx_RM = antisym(jmax, True)

    # Record run infomation
    text = open('run_info(Nt=%s,tmax=%s)' %(imax,pm.tmax), 'w')
    text.write("Information:\n\n")
    text.write("jmax:%g\nkmax:%g\nimax:%g\nxmax:%g\ntmax:%g\n" %(jmax,kmax,imax,xmax,pm.tmax))
    text.write("deltax:%g\ndeltat:%g\n" %(deltax,pm.deltat))
    text.close()

    # Complex Time array initialisations 
    string = 'Many Body Complex Time: Constructing Arrays'
    sprint.sprint(string,2,0,msglvl)
    sprint.sprint(string,1,0,msglvl)
    Psiarr = np.zeros((2,jmax**2), dtype = np.cfloat)
    Mat = sparse.lil_matrix((jmax**2,jmax**2),dtype = np.cfloat)
    Rhv2 = np.zeros((jmax**2), dtype = np.cfloat)
    Psi2D = np.zeros((jmax,kmax), dtype = np.cfloat)
    r = 0.0 + (1.0)*(deltat/(4.0*(deltax**2))) 

    # Evolve throught complex time
    CNsolveComplexTime() 

    # Real Time array initialisations 
    string = 'Many Body Real Time: Constructing Arrays'
    sprint.sprint(string,1,0,msglvl)
    sprint.sprint(string,2,0,msglvl)
    Psiarr = np.zeros((2,jmax**2), dtype = np.cfloat)
    Psi2D = np.zeros((jmax,kmax), dtype = np.cfloat)
    Rhv2 = np.zeros((jmax**2), dtype = np.cfloat)
    Mat2 = sparse.lil_matrix((jmax**2,jmax**2),dtype = np.cfloat)

    # Evolve throught real time
    if int(TD) == 1:
        tmax = pm.tmax
        imax = pm.imax
        deltat = tmax/(imax-1)
    if int(TD) == 0:
       tmax = 0.0
       imax = 1
       deltat = 0.0
    deltax = pm.deltax
    r = 0.0 + (1.0j)*(deltat/(4.0*(deltax**2)))
    CNsolveRealTime()

# Run stand-alone
if(__name__ == '__main__'):
    main()

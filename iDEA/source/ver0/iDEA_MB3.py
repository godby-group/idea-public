######################################################################################
# Name: 3 electron Many Body                                                         #
######################################################################################
# Author(s): Jacob Chapman, Thomas Durrant, Jack Wetherell                           #
######################################################################################
# Description:                                                                       #
# Computes exact many body wavefunction and density                                  #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################

# Do not run stand-alone
if(__name__ == '__main__'):
    print('do not run stand-alone')
    quit()

# Library Imports
import sys
import mkl
import time
import math
import copy
import cPickle  
import pickle
import sprint
import numpy as np
import scipy as sp
import os as os
import fort as f
from scipy import linalg as la
from scipy import special
from scipy import sparse
from antisym3e import *
from scipy.sparse import linalg as spla
import parameters as pm
from ses import get_MB_groundstDen as groundstDen

# Variable initialisation
jmax = pm.jmax
kmax = pm.kmax
lmax = pm.lmax
imax = pm.imax
cimax = pm.cimax
xmax = pm.xmax
tmax = pm.tmax
ctmax = pm.ctmax
deltax = pm.deltax
deltat = tmax/(imax-1)
cdeltat = ctmax/(cimax-1)
antifact = pm.antifact
ctol = pm.ctol
rtol = pm.rtol
TD = pm.TD
par = pm.par
msglvl = pm.msglvl
gdstD = 0
c_m = 0
c_p = 0 
Nx_RM = 0

# Takes every combination of the two electron indicies and creates a single unique index 
def Gind(k,j,l):
    return (k + j*jmax + l*(jmax**2))
    
# Inverses the Gind operation. Takes the single index and returns the corresponding indices used to create it.
def InvGind(jkl):
    k = jkl % jmax
    j = ((jkl - k)%(jmax**2))/jmax
    l = (jkl - k - j*jmax)/(jmax**2)
    return k, j, l

# Calculates the nth Energy Eigenfunction of the Harmonic Oscillator (~H(n)(x)exp(x^2/2))
def EnergyEigenfunction(n):
    j = 0
    x = -xmax
    Psi = np.zeros(jmax, dtype = np.cfloat)
    while (x < xmax):
        factorial = np.arange(0, n+1, 1)
        fact = np.product(factorial[1:])
        norm = (np.sqrt(1.0/((2.0**n)*fact)))*((1.0/math.pi)**0.25)
        Psi[j] = complex(norm*(sp.special.hermite(n)(x))*(1**2)*np.exp(-0.5*(x**2)*(1**2)), 0.0)
        j = j + 1
        x = x + deltax
    return Psi

# Builds the Cayley Matrix on the LHS (C1) of the matrix eigenproblem in Complex Time (t*=-it): C1=(I+H.dt/2)
def Matrixdef(i,r):
    j = 0
    k = 0
    l = 0
    while (l < lmax):
        j = 0
        while (j < jmax):
            k=0
            while (k < kmax):
                jkl = Gind(k,j,l)
                a=0;b=0;c=0;d=0;e=0;f=0
                Mat[jkl,jkl] = 1.0 + (6.0*r) + (2.0*r*(deltax**2)*(pm.Potential3(i,k,j,l)))
                aa=Mat[jkl,jkl]
                if (k<kmax-1):
                    if ((Gind(k+1,j,l) >= 0) and (Gind(k+1,j,l) < jmax**3)):
                        Mat[jkl,Gind(k+1,j,l)] = -r 
                if (k>0):
                    if (Gind(k-1,j,l) >= 0 and Gind(k-1,j,l) < jmax**3):
                        Mat[jkl,Gind(k-1,j,l)] = -r
                if (j <= jmax - 1):
                    if (Gind(k,j+1,l) >= 0 and Gind(k,j+1,l) < jmax**3):
                        Mat[jkl,Gind(k,j+1,l)] = -r
                if (j>=0):
                    if (Gind(k,j-1,l) >= 0 and Gind(k,j-1,l) < jmax**3):
                        Mat[jkl,Gind(k,j-1,l)] = -r
                if (l < lmax - 1):
                    if (Gind(k,j,l+1) >= 0 and Gind(k,j,l+1) < jmax**3):
                        Mat[jkl,Gind(k,j,l+1)] = -r
                if (l > 0):
                    if (Gind(k,j,l-1) >= 0 and Gind(k,j,l-1) < jmax**3):
                        Mat[jkl,Gind(k,j,l-1)] = -r
                k = k + 1
            j = j + 1
        l = l + 1
    return Mat

# Imaginary Time Crank Nicholson initial condition
def InitialconI():
    Psi1 = np.zeros(jmax,dtype = np.cfloat)
    Psi2 = np.zeros(kmax,dtype = np.cfloat)
    Psi3 = np.zeros(lmax,dtype = np.cfloat)
    Psi1 = EnergyEigenfunction(0)
    Psi2 = EnergyEigenfunction(1)
    Psi3 = EnergyEigenfunction(2)
    l = 0
    while (l < lmax):
        j = 0
        while (j < jmax):
            k = 0
            while (k < kmax):
                Paulix = Psi1[k]*(Psi2[j]*Psi3[l]-Psi2[l]*Psi3[j])
                Pauliy = Psi2[k]*(Psi1[j]*Psi3[l]-Psi1[l]*Psi3[j])
                Pauliz = Psi3[k]*(Psi1[j]*Psi2[l]-Psi1[l]*Psi2[j])
                Pauli = Paulix - Pauliy + Pauliz 
                Psiarr[0,Gind(k,j,l)] = Pauli
                k = k + 1
            j = j + 1
        l = l + 1
    return Psiarr[0,:]

# Takes the ground state solution and uses it as the initial eigenfunction in the real propagation
def InitialconR():
    Psi3D = np.zeros((lmax,jmax,kmax), dtype = np.cfloat)
    Psi3D=np.load('Psi_IC_Real3D.npy')
    Psiarr[0,:] = PsiInverter(Psi3D[:,:,:],0)
    return Psiarr[0,:]

# Define function to turn array of compressed indexes into seperated indexes
def PsiConverterI(Psiarr,i):
    Psi3D = np.zeros((jmax,kmax,lmax), dtype = np.cfloat)
    #origdir = os.getcwd()
    #newdir = os.path.join('MPsiReal_binaries_3D')
    #if not os.path.isdir(newdir):
    #    os.mkdir(newdir)
    jkl = 0
    while (jkl < jmax**3):
        k, j, l = InvGind(jkl)
        Psi3D[k,j,l] = Psiarr[jkl]
        jkl = jkl + 1
    # Save the wavefunction for the ith time step:
    #        Psi3D = three electron density array
    #        mPsi3D = modulus of above (Psi*.Psi)
    # Saved into the directory MPsiReal_binaries_3D
    # Save at every timestep
    #if (i > 0):  
    #    #mPsi3D[k,j,l] = (np.absolute(Psi3D[k,j,l]**2)) 
    #    os.chdir(newdir)
    #    np.save('Psi2DComplex_%i' %(i),Psi3D)
    #    os.chdir(origdir)
    if (i!=0):
        np.save('Psi_IC_Real', Psiarr[:])
        np.save('Psi_IC_Real3D', Psi3D)
        np.save('Psi2DComplex_%i' %(i),Psi3D)
    if (i==0):
        np.save('Psi_IC_Inital', Psiarr[:])
        np.save('Psi_IC_Inital3D', Psi3D)
        np.save('Psi2DComplex_%i' %(i),Psi3D)  
    return 

# Cut down version that returns Psi3D and does not write to file - works for both times
def PsiConverterLite(Psiarr):
    Psi3D = np.zeros((jmax,kmax,lmax), dtype = np.cfloat)
    jkl = 0
    while (jkl < jmax**3):
        k, j, l = InvGind(jkl)
        Psi3D[k,j,l] = Psiarr[jkl]
        jkl = jkl + 1
    return Psi3D
 
# Output the 3D wavefunction to file during the real time evolution
def PsiConverterR(Psiarr,i):
    Psi3D = np.zeros((jmax,kmax,lmax), dtype = np.cfloat)
    mPsi3D = np.zeros((jmax,kmax,lmax))
    jkl = 0
    while (jkl < jmax**3):
        k, j, l = InvGind(jkl)
        Psi3D[k,j,l] = Psiarr[jkl]
        mPsi3D[k,j,l] = (np.absolute(Psi3D[k,j,l])**2)
        jkl = jkl + 1                                         
    den = np.zeros(jmax, dtype=np.float)
    for element in range(0, jmax):
        den[element] = 3*np.sum(mPsi3D[element,:,:])*deltax**2
    np.save('chargeDensity_%i' %(i),den)
    return

# Psi inverter
def PsiInverter(Psi3D,i):
    Psiarr = np.zeros((jmax**3), dtype = np.cfloat)
    j = 0
    k = 0
    l = 0    
    while (l < lmax):
        j = 0
        while (j < jmax):
            k = 0
            while (k < kmax):            
                jkl = Gind(k,j,l)
                Psiarr[jkl] = Psi3D[k,j,l]
                k = k + 1
            j = j + 1
        l = l + 1
    return Psiarr[:]
	
# Calculate the energy from present and previous psi during complex time
def Energy(Psi):
    a = np.linalg.norm(Psi[0,:])
    b = np.linalg.norm(Psi[1,:])
    return -(np.log(b/a))/cdeltat

# Function to output the system's external potential
def OutputPotential():
    output_file1 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_2gs_ext_vxt.db','w')
    potential = []
    i = 0
    while(i < pm.grid):
        potential.append(pm.well(float(i*pm.deltax)-pm.xmax))
        i = i + 1
    pickle.dump(potential,output_file1)
    output_file1.close()
    if(pm.TD == 1):
        output_file2 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_2td_ext_vxt.db','w')
        TDP = []
        i = 0
        while(i < pm.imax):
            TDP.append(potential)
            i = i + 1
        pickle.dump(TDP,output_file2)
        output_file2.close()
    return

# Function to construct the real matrix Af 
def ConstructAf(A):
    A1_dat, A2_dat = mkl.mkl_split(A.data,len(A.data))
    A.data = A1_dat
    A1 = copy.copy(A)
    A.data = A2_dat
    A2 = copy.copy(A)
    Af = sp.sparse.bmat([[A1,-A2],[A2,A1]]).tocsr()
    return Af

# Apply the Crank-Nickolson method using complex time to find the ground state of the system
def CNsolveComplexTime():
    i = 1

    # Check if we have a known groundstate density to calculate errors from
    try:
        gdstD
    except NameError:
        errors = False
    else:
        errors = True

    # Set the initial condition of the wavefunction
    Psiarr[0,:] = InitialconI()
    Psiarr_RM = c_m*Psiarr[0,:]

    # Construct the matrix A
    A = Matrixdef(0,r)
    A = A.tocsc()
    A_RM = c_m*A*c_p

    # Construct the matrix C
    C = -(A-sp.sparse.identity(jmax**3, dtype=np.cfloat))+sp.sparse.identity(jmax**3, dtype=np.cfloat)
    C_RM = c_m*C*c_p

    # Perform iterations
    while (i < cimax):

	# Begin timing the iteration
        start = time.time()
        string = 'Complex Time = ' + str(i*deltat)
	sprint.sprint(string,2,0,msglvl)

        # Reduce the wavefunction
        if (i==1):
            PsiConverterI(Psiarr[0,:],0)
        elif (i>=2):
            Psiarr[0,:]=Psiarr[1,:]
   
	# Construct vector b
	if(par == 0):
	    b_RM = C_RM*Psiarr_RM
	else:
	    b_RM = mkl.mkl_mvmultiply_c(C_RM.data,C_RM.indptr+1,C_RM.indices+1,1,Psiarr_RM,C_RM.shape[0],C_RM.indices.size)

	# Solve Ax=b
	Psiarr_RM,info = spla.minres(A_RM,b_RM,x0=Psiarr_RM,tol=ctol)
        
	# Expand the wavefunction
        Psiarr[1,:] = c_p*Psiarr_RM
        
	# Calculate the energy
        Ev = Energy(Psiarr)
	string = 'Energy = ' + str(Ev)
	sprint.sprint(string,2,0,msglvl)
        
	# Normalise new wavefunction
        mag = np.sqrt((np.sum(Psiarr[1,:]*np.conjugate(Psiarr[1,:]))*(deltax**3)))
        Psiarr[1,:] = Psiarr[1,:] / mag
        Psiarr_RM = c_m*Psiarr[1,:]
        
	# Stop timing the iteration
	finish = time.time()
        string = 'Time to Complete Step: ' + str(finish-start)
	sprint.sprint(string,2,0,msglvl)

	# Test for convergance
	wf_con = np.linalg.norm(Psiarr[0,:]-Psiarr[1,:])
	string = 'wave function convergence: ' + str(wf_con)
	sprint.sprint(string,2,0,msglvl)
	string = 'many body complex time: ' + 't = ' + str(i*deltat) + ', convergence = ' + str(wf_con)
        sprint.sprint(string,1,1,msglvl)
	if(i>1):
	    e_con = old_energy - Ev
	    string = 'energy convergence: ' + str(e_con)
	    sprint.sprint(string,2,0,msglvl)
	    normal = np.sum(np.absolute(Psiarr[1,:])**2)*(deltax**3)          
            string = 'normalisation: ' + str(normal)                          
            sprint.sprint(string,2,0,msglvl)                                  
            string = 'residual: ' + str(np.linalg.norm(A_RM*Psiarr_RM-b_RM))  
	    sprint.sprint(string,2,0,msglvl)                                  
	    if(wf_con < ctol*10.0):
		print
	        string = 'many body complex time: ground state converged' 
		sprint.sprint(string,1,0,msglvl)
                string = 'ground state converged' 
		sprint.sprint(string,2,0,msglvl)
	        i = cimax
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
    C = -(A-sp.sparse.identity(jmax**3, dtype=np.cfloat))+sp.sparse.identity(jmax**3, dtype=np.cfloat)
    C_RM = c_m*C*c_p
    while (i < imax):
	# Begin timing the iteration
        start = time.time()
	string = 'Real Time = ' + str(i*deltat) + '/' + str((imax-1)*deltat)
	sprint.sprint(string,2,0,msglvl)

        if (i>=2):
            Psiarr_RM = c_m*Psiarr[1,:]    
        
        # Construct the vector b
	b = C*Psiarr[0,:]
        if(par == 0):
	    b_RM = C_RM*Psiarr_RM
	else:
	    b_RM = mkl.mkl_mvmultiply_c(C_RM.data,C_RM.indptr+1,C_RM.indices+1,1,Psiarr_RM,C_RM.shape[0],C_RM.indices.size)
        
	# Solve Ax=b
	if(par == 0):
	    Psiarr_RM,info = spla.lgmres(A_RM,b_RM,x0=Psiarr_RM,tol=rtol)
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
        text = '\n time Step =' + str(i) + ', integral of modulus over all space, ie. P(particle1 and particle2)='
	text = text + str(np.sum(np.absolute(Psiarr[1,:])**2)*(deltax**2))
        Particle.write(text)

        # Stop timing the iteration
	finish = time.time()
        string = 'time to complete step: ' + str(finish-start)
	sprint.sprint(string,2,0,msglvl)

        # Print to screen
        string = 'residual: ' + str(np.linalg.norm(A_RM*Psiarr_RM-b_RM))
	sprint.sprint(string,2,0,msglvl)
	normal = np.sum(np.absolute(Psiarr[1,:])**2)*(deltax**3)
        string = 'normalisation: ' + str(normal)
        sprint.sprint(string,2,0,msglvl)
        antisym =  np.sum(abs(f.elec3.antisym1d(Psiarr[1:,], jmax))**2) * deltax * (1.0/6.0**2) 
	string = 'many body real time: ' + 't = ' + str(i*deltat) + ', normalisation = ' + str(normal)
        sprint.sprint(string,1,1,msglvl)
        string = '---------------------------------------------------'
	sprint.sprint(string,2,0,msglvl)

	# Iterate
        i += 1

    # Dispose of matrices and terminate
    A = 0
    C = 0
    Particle.close()
    return

# Call this function to run iDEA-MB for 3 electrons
def main():

    # Use global variables
    global jmax,kmax,lmax,xmax,tmax,deltax,deltat,gdstD,imax,msglvl,Psiarr,Mat,Rhv2,Psi3D,r,Mat3,c_m,c_p
    string = 'many body complex time: constructing arrays'
    sprint.sprint(string,2,0,msglvl)
    sprint.sprint(string,1,0,msglvl)
    OLD = os.getcwd()
    os.chdir('outputs/' + str(pm.run_name) + '/')

    # Construct reduction and expansion matrices
    c_m, c_p = antisym(jmax)

    # Complex Crank Nicholoson Array Initialisations
    Psiarr = np.zeros((2,jmax**3), dtype = np.cfloat)   
    Mat = sparse.lil_matrix((jmax**3,jmax**3),dtype = np.cfloat)	
    Mat3 = sparse.lil_matrix((jmax**3,jmax**3),dtype = np.cfloat)  
    Rhv2 = np.zeros((jmax**3), dtype = np.cfloat)      
    Psi3D = np.zeros((kmax,jmax,lmax), dtype = np.cfloat)  
    r = 0.0 + (1.0)*(deltat/(4.0*(deltax**2))) 
    gdstD = groundstDen()

    # Complex time evolution
    Psi3D_Int = CNsolveComplexTime()

    # Real Time CN array initialisations
    if int(TD) == 1:
        string = 'many body real time: constructing arrays'
        sprint.sprint(string,1,0,msglvl)
        sprint.sprint(string,2,0,msglvl)
        Psiarr = np.zeros((2,jmax**3), dtype = np.cfloat)	     
        Rhv2 = np.zeros((jmax**3), dtype = np.cfloat)     
        Mat2 = sparse.lil_matrix((jmax**3,jmax**3),dtype = np.cfloat)
        r = 0.0 + (1.0j)*(deltat/(4.0*(deltax**2)))

    # Evolve throught real time
    if int(TD) == 1:
    	CNsolveRealTime()
        Den = np.zeros((imax,jmax))
        for i in range(imax):     
            Den[i,:] = np.load('chargeDensity_%i.npy' %(i))
        ProbPsiFile = open('raw/' + str(pm.run_name) + '_3td_ext_den.db','w')
        pickle.dump(Den, ProbPsiFile)
        ProbPsiFile.close()
    imax = 1
    CNsolveRealTime()
    Den = np.zeros(jmax)
    Den[:] = np.load('chargeDensity_%i.npy' %(0))
    os.remove('chargeDensity_%i.npy'%(0)) 
    ProbPsiFile = open('raw/' + str(pm.run_name) + '_3gs_ext_den.db','w')
    pickle.dump(Den, ProbPsiFile)
    ProbPsiFile.close()
    OutputPotential()
    os.chdir(OLD)
    print


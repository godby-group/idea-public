######################################################################################
# Name: 2 electron Many Body                                                         #
######################################################################################
# Author(s): Jack Wetherell, James Ramsden                                           #
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
import os
import mkl
import time
import math
import copy
import pickle
import sprint
import numpy as np
import scipy as sp
import RE_Utilities
import parameters as pm
from scipy import sparse
from scipy import special
from scipy.misc import factorial
import scipy.sparse.linalg as spla

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

# Define potential array for all spacial points
def Potential(i,j,k):
    V = np.zeros((pm.jmax,pm.kmax))
    xk = -pm.xmax + (k*pm.deltax)
    xj = -pm.xmax + (j*pm.deltax)
    if (i == 0): 	
        V[j,k] = pm.well(xk) + pm.well(xj) + pm.inte*(1.0/(abs(xk-xj) + pm.acon))
    else:
        V[j,k] = pm.well(xk) + pm.well(xj) + pm.inte*(1.0/(abs(xk-xj) + pm.acon)) + pm.petrb(xk) + pm.petrb(xj)
    return V[j,k]

# Builds the Cayley Matrix on the LHS (C1) of the matrix eigenproblem in Complex Time (t*=-it): C1=(I+H.dt/2)
def Matrixdef(i,r):
    j = 0
    k = 0
    while (j < jmax):
        k = 0
        while (k < kmax):
            jk = Gind(j,k)
            a=0;b=0;c=0;d=0
            Mat[jk,jk] = 1.0 + (4.0*r) + (2.0*r*(deltax**2)*(Potential(i,j,k)))
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
    return mPsi2D

# Define function to turn array of compressed indexes into seperated indexes
def PsiConverterR(Psiarr):
    Psi2D = np.zeros((jmax,kmax), dtype = np.cfloat)
    mPsi2D = np.zeros((jmax,kmax))
    jk = 0
    while (jk < jmax**2):
        j, k = InvGind(jk)
        Psi2D[j,k] = Psiarr[jk]
        jk = jk + 1
    mPsi2D[:,:] = (np.absolute(Psi2D[:,:])**2)
    return mPsi2D

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

# Function to calculate the current density
def CalculateCurrentDensity(n,i):	
    J=np.zeros(pm.jmax)
    RE_Utilities.continuity_eqn(i+1,pm.jmax,pm.deltax,pm.deltat,n,J)
    return J

# Conctruct grid for antisym matrices
def gridz(N_x):
    N_e = 2
    Nxl = N_x**N_e
    Nxs = np.prod(range(N_x,N_x+N_e))/int(factorial(N_e))
    sgrid = np.zeros((N_x,N_x), dtype='int')
    count = 0
    for ix in range(N_x):
        for jx in range(ix+1):
            sgrid[ix,jx] = count
            count += 1
    lgrid = np.zeros((N_x,N_x), dtype='int')
    lky = np.zeros((Nxl,2), dtype='int')
    count = 0
    for ix in range(N_x):
        for jx in range(N_x):
            lgrid[ix,jx] = count
            lky[count,0] = ix
            lky[count,1] = jx
            count += 1
    return sgrid, lgrid

# Construct antisym matrices
def antisym(N_x, retsize=False):
    N_e = 2
    Nxl = N_x**N_e
    Nxs = np.prod(range(N_x,N_x+N_e))/int(factorial(N_e))
    sgrid, lgrid = gridz(N_x)
    C_down = sparse.lil_matrix((Nxs,Nxl))
    for ix in range(N_x):
        for jx in range(ix+1):
            C_down[sgrid[ix,jx],lgrid[ix,jx]] = 1.
    C_up = sparse.lil_matrix((Nxl,Nxs))
    for ix in range(N_x):
        for jx in range(N_x):
            il = lgrid[ix,jx]
            ish = sgrid[ix,jx]
            if jx <= ix:
                C_up[il,ish] = 1.
            else:
                jsh = sgrid[jx,ix]
                C_up[il,jsh] = -1.
    C_down = C_down.tocsr()
    C_up = C_up.tocsr()
    if retsize:
        return C_down, C_up, Nxs
    else:
        return C_down, C_up

# Function to output the system's external potential
def OutputPotential():
    output_file1 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_2gs_ext_vxt.db','w')
    potential = []
    i = 0
    while(i < pm.grid):
        potential.append(pm.well(float((i*deltax)-pm.xmax)))
        i = i + 1
    pickle.dump(potential,output_file1)
    output_file1.close()
    if(pm.TD == 1):
        potential2 = []
        i = 0
        while(i < pm.grid):
        	potential2.append(pm.well(float((i*deltax)-pm.xmax)) + pm.petrb(float((i*deltax)-pm.xmax)))
        	i = i + 1
        output_file2 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_2td_ext_vxt.db','w')
        TDP = []
        i = 0
        while(i < pm.imax):
            TDP.append(potential2)
            i = i + 1
        pickle.dump(TDP,output_file2)
        output_file2.close()
    return

# Function to calculate the current density
def calculateCurrentDensity(total_td_density):
    current_density = []
    for i in range(0,len(total_td_density)-1):
         string = 'MB: computing time dependent current density t = ' + str(i*pm.deltat)
         sprint.sprint(string,1,1,pm.msglvl)
         J = np.zeros(pm.jmax)
         J = RE_Utilities.continuity_eqn(pm.jmax,pm.deltax,pm.deltat,total_td_density[i+1],total_td_density[i])
         current_density.append(J)
    return current_density

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
        string = 'complex time = ' + str(i*deltat)
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
	string = 'energy = ' + str(Ev)
	sprint.sprint(string,2,0,msglvl)

	# Normalise the wavefunction
	mag = (np.linalg.norm(Psiarr[1,:])*deltax)
        Psiarr[1,:] = Psiarr[1,:]/mag
	
	# Stop timing the iteration
	finish = time.time()
        string = 'time to Complete Step: ' + str(finish-start)
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
	    if(e_con < ctol*10.0 and wf_con < ctol*10.0):
		print
	        string = 'many body complex time: ground state converged' 
		sprint.sprint(string,1,0,msglvl)
                string = 'ground state converged' 
		sprint.sprint(string,2,0,msglvl)
	        i = imax
	old_energy = copy.copy(Ev)
        string = '---------------------------------------------------'
	sprint.sprint(string,2,0,msglvl)

        # Iterate
        i += 1

    # Convert Psi
    Psi2Dcon = PsiConverterI(Psiarr[1,:],i)
 
    # Calculate denstiy
    density = np.sum(Psi2Dcon[:,:], axis=0)*deltax*2.0

    # Output ground state density
    output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_2gs_ext_den.db','w')
    pickle.dump(density,output_file)
    output_file.close()
    OutputPotential()

    # Dispose of matrices and terminate
    A = 0
    C = 0
    return Psiarr[1,:]

# Function to iterate over real time
def CNsolveRealTime(wavefunction):
    i = 1

    # Initialse wavefunction
    Psiarr[0,:] = wavefunction
    PsiConverterR(Psiarr[0,:])
    Psiarr_RM = c_m*Psiarr[0,:]

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
    TDD = []
    TDD_GS = []
    current_density = []

    # Save ground state
    GS = np.sum(PsiConverterR(wavefunction), axis=0)*deltax*2.0
    TDD_GS.append(GS)

    while (i <= imax):

	# Begin timing the iteration
        start = time.time()
	string = 'real time = ' + str(i*deltat) + '/' + str((imax-1)*deltat)
	sprint.sprint(string,2,0,msglvl)

	# Reduce the wavefunction
        if (i>=2):
            Psiarr[0,:] = Psiarr[1,:]
            Psiarr_RM = c_m*Psiarr[0,:]

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
        Psi2Dcon = PsiConverterR(Psiarr[1,:])

        # Calculate denstiy
        density = np.sum(Psi2Dcon[:,:], axis=0)*deltax*2.0
        TDD.append(density)
        TDD_GS.append(density)

	# Stop timing the iteration
	finish = time.time()
        string = 'Time to Complete Step: ' + str(finish-start)
	sprint.sprint(string,2,0,msglvl)

	# Print to screen
        string = 'residual: ' + str(np.linalg.norm(A*Psiarr[1,:]-b))
	sprint.sprint(string,2,0,msglvl)
	normal = np.sum(np.absolute(Psiarr[1,:])**2)*(deltax**2)
        string = 'normalisation: ' + str(normal)
	sprint.sprint(string,2,0,msglvl)
	string = 'many body real time: ' + 't = ' + str(i*deltat) + ', normalisation = ' + str(normal)
        sprint.sprint(string,1,1,msglvl)
        string = '---------------------------------------------------'
	sprint.sprint(string,2,0,msglvl)

	# Iterate
        i += 1

    # Calculate current density
    current_density = calculateCurrentDensity(TDD_GS)

    # Output time dependent density
    output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_2td_ext_den.db','w')
    pickle.dump(TDD,output_file)

    # Output time dependent current density
    output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_2td_ext_cur.db','w')
    pickle.dump(current_density,output_file)

    # Dispose of matrices and terminate
    A = 0
    C = 0
    sprint.sprint(' ',1,0,msglvl)
    return

# Call this function to run iDEA-MB for 2 electrons
def main():

    # Use global variables
    global jmax,kmax,xmax,tmax,deltax,deltat,imax,msglvl,Psiarr,Mat,Rhv2,Psi2D,r,Mat2,c_m,c_p,Nx_RM

    # Construct reduction and expansion matrices
    c_m, c_p, Nx_RM = antisym(jmax, True)

    # Complex Time array initialisations 
    string = 'many body complex time: constructing arrays'
    sprint.sprint(string,2,0,msglvl)
    sprint.sprint(string,1,0,msglvl)
    Psiarr = np.zeros((2,jmax**2), dtype = np.cfloat)
    Mat = sparse.lil_matrix((jmax**2,jmax**2),dtype = np.cfloat)
    Rhv2 = np.zeros((jmax**2), dtype = np.cfloat)
    Psi2D = np.zeros((jmax,kmax), dtype = np.cfloat)
    r = 0.0 + (1.0)*(deltat/(4.0*(deltax**2))) 

    # Evolve throught complex time
    wavefunction = CNsolveComplexTime() 

    # Real Time array initialisations 
    if(pm.TD == 1):
        string = 'many body real time: constructing arrays'
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
        deltax = pm.deltax
        r = 0.0 + (1.0j)*(deltat/(4.0*(deltax**2)))
        CNsolveRealTime(wavefunction)
    if int(TD) == 0:
       tmax = 0.0
       imax = 1
       deltat = 0.0


######################################################################################
# Name: Reverse Engineering                                                          #
#                                                                                    #
######################################################################################
# Authors: Matt Hodgson, James Ramsden and Matthew Smith                             #
#                                                                                    #
######################################################################################
# Description:                                                                       #
# Computes exact VKS, VH, VXC using the RE algorithm from the exact density          #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
# Ground-state calculations are usually fast and stable, this may vary if the system #
# is particularly difficult to RE, i.e. if the system has regions of very small      #
# electron density. To control the rate of convergence, and the stability of the     #
# GSRE, use the variables 'mu' and 'p'. p is used for the rate of convergence        #
# and for bringing more attention to regions of low density. mu is used for          #
# stabiliesing the algorithm. Time-dependent RE is much more difficult, and is       #
# dependent on the system. If the TDRE is not converging the most likely reason      #
# is that dt is too big. There could also be a problem with noise. Noise should be   #
# obvious in he velocity field (current/density). If noise is dominating the system, #
# try changing the noise filtering value 'alpha'. Alpha controls how much of the     #
# high frequency terms are removed from the KS vector potential.                     #
#                                                                                    #
######################################################################################

# Import librarys
import os
import sys
import copy
import time
import math
import pickle
import sprint
import numpy as np
import RE_Utilities
import parameters as pm
from scipy import sparse
from scipy import special
from scipy import linalg as la
from scipy.sparse import linalg as spla									
from scipy.linalg import eig_banded, solve

# Function to read inputs
def ReadInput(approx,n_MB,GS,imax):
    if pm.TD==1:
        if GS==0:
            # Read in the ground-state first
            file_name='outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_den.db'
            input_file=open(file_name,'r')
            data=pickle.load(input_file)
            n_MB[0,:]=data
        else:
            Read_n_MB=np.zeros(((imax-1),pm.jmax),dtype='float')

            # Then read im the time-dependent density
            file_name='outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_den.db'
            input_file=open(file_name,'r')
            data=pickle.load(input_file)
            Read_n_MB[:,:]=data
            for k in range(1,imax):
                n_MB[k,:]=Read_n_MB[k-1,:] # Accounts for the difference in convention between MB and RE (for RE t=0 is the ground-state)
    if pm.TD==0:
        # Only a ground-state to read in
        file_name='outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_den.db'
        input_file=open(file_name,'r')
        data=pickle.load(input_file)
        n_MB[0,:]=data
    return n_MB

# Function to calculate the ground-state potential
def CalculateGroundstate(V_KS,n_MB,mu,sqdx,V_ext,T,n_KS):

    #Build Hamiltonian
    p=0.05 # Determines the rate of convergence of the ground-state RE
    HGS=copy.copy(T)
    V_KS[0,:]+=mu*(n_KS[0,:]**p-n_MB[0,:]**p)
    HGS[0,:]+=V_KS[0,:]

    # Solve KS equations
    K,U=eig_banded(HGS,True)
    Psi = np.zeros((pm.NE,2,pm.jmax), dtype='complex')
    for i in range(pm.NE):
        Psi[i,0,:] = U[:,i]/sqdx # Normalise

    # Calculate density and cost function
    n_KS[0,:]=0
    for i in range(pm.NE):
        n_KS[0,:]+=abs(Psi[i,0,:])**2 # Calculate the density from the single-particle wavefunctions
    cost_n_GS=sum(abs(n_MB[0,:]-n_KS[0,:]))*pm.deltax # Calculate the ground-state cost function 
    return V_KS,n_KS,cost_n_GS,Psi

# Function to load or force calculation of the ground-state potential
def GroundState(V_KS,n_MB,mu,sqdx,V_ext,T,n_KS):
    print 'REV: calculating ground-state Kohn-Sham potential'
    for i in range(pm.jmax):
        V_KS[0,i]=pm.well((i*pm.deltax-pm.xmax)) # Initial guess for KS potential
        V_ext[i]=pm.well((i*pm.deltax-pm.xmax))
    V_KS,n_KS,cost_n_GS,U=CalculateGroundstate(V_KS,n_MB,0,sqdx,V_ext,T,n_KS)
    print 'REV: initial guess electron density error = %s' % cost_n_GS
    while cost_n_GS>1e-13:
        cost_old = cost_n_GS
        string = 'REV: electron density error = ' + str(cost_old)
	sprint.sprint(string,1,1,pm.msglvl)
	sprint.sprint(string,2,1,pm.msglvl)
        V_KS,n_KS,cost_n_GS,U=CalculateGroundstate(V_KS,n_MB,mu,sqdx,V_ext,T,n_KS)
	if abs(cost_n_GS-cost_old)<1e-15 or cost_n_GS>cost_old:
	    mu*=0.5
        if mu < 1e-15:
            break
    return V_KS,n_KS,U,V_ext

# Function used in calculation of the Hatree potential
def realspace(vector):
    mid_k=int(0.5*(pm.jmax-1))
    fftin=np.zeros(pm.jmax-1,dtype='complex')
    fftin[0:mid_k+1]=vector[mid_k:pm.jmax]
    fftin[pm.jmax-mid_k:pm.jmax-1]=vector[1:mid_k]
    fftout=np.fft.ifft(fftin)
    func=np.zeros(pm.jmax, dtype='complex')
    func[0:pm.jmax-1]=fftout[0:pm.jmax-1]
    func[pm.jmax-1]=func[0]
    return func

# Function used in calculation of the Hatree potential
def momentumspace(func):
    mid_k=int(0.5*(pm.jmax-1))
    fftin=np.zeros(pm.jmax-1,dtype='complex')
    fftin[0:pm.jmax-1] = func[0:pm.jmax-1] + 0.0j
    fftout=np.fft.fft(fftin)
    vector=np.zeros(pm.jmax,dtype='complex')
    vector[mid_k:pm.jmax]=fftout[0:mid_k+1]
    vector[1:mid_k]=fftout[pm.jmax-mid_k:pm.jmax-1]
    vector[0]=vector[pm.jmax-1].conjugate()
    return vector

# Function used to calculate the Hatree potential
def Hartree(n):
    n_k=momentumspace(n)*pm.deltax
    X_x=np.zeros(pm.jmax)
    for i in range(pm.jmax):
        X_x[i]=1.0/(abs(i*pm.deltax-pm.xmax)+pm.acon)
    X_k=momentumspace(X_x)*pm.deltax/(2*pm.xmax)
    V_k=np.zeros(pm.jmax,dtype='complex')
    V_k[:]=X_k[:]*n_k[:]
    fftout=realspace(V_k).real*2*pm.xmax/pm.deltax
    V_hx=np.zeros(pm.jmax)
    V_hx[0:0.5*(pm.jmax+1)]=fftout[0.5*(pm.jmax-1):pm.jmax]
    V_hx[0.5*(pm.jmax+1):pm.jmax-1]=fftout[1:0.5*(pm.jmax-1)]
    V_hx[pm.jmax-1]=V_hx[0]
    return V_hx

# Function to extrapolate the current density from regions of low density to the system's edges
def ExtrapolateCD(J,j,n,n_MB,upper_bound):
    imaxl=0 # Start from the edge of the system
    nmaxl=0.0
    imaxr=0									
    nmaxr=0.0
    for l in range(upper_bound+1):
        if n_MB[j,l]>nmaxl: # Find the first peak in the density from the left
            nmaxl=n_MB[j,l]
	    imaxl=l
        i=upper_bound+l-1
        if n_MB[j,i]>nmaxr: # Find the first peak in the density from the right
             nmaxr=n_MB[j,i]
             imaxr=l
    U=np.zeros(pm.jmax)
    U[:]=J[:]/n[j,:]
    dUdx=np.zeros(pm.jmax)

    # Extraplorate the density for the low density regions
    for i in range(imaxl+1):
        l=imaxl-i
        if n_MB[j,l]<1e-8:
            dUdx[:]=np.gradient(U[:],pm.deltax)
            U[l]=8*U[l+1]-8*U[l+3]+U[l+4]+dUdx[l+2]*12.0*pm.deltax
    for i in range(int(0.5*(pm.jmax-1)-imaxr+1)):
        l=int(0.5*(pm.jmax-1)+imaxr+i)
        if n_MB[j,l]<1e-8:
            dUdx[:]=np.gradient(U[:],pm.deltax)
            U[l]=8*U[l-1]-8*U[l-3]+U[l-4]-dUdx[l-2]*12.0*pm.deltax
    J[:]=n[j,:]*U[:]							
    return J

# Function to extrapolate the KS vector potential from regions of low density to the system's edges
def ExtrapolateVectorPotential(A_KS,n_MB,j,upper_bound):
    imaxl=0
    nmaxl=0.0
    imaxr=0									
    nmaxr=0.0
    for i in range(upper_bound+1):
        if n_MB[j,i]>nmaxl:
            nmaxl=n_MB[j,i]
            imaxl=i
    for l in range(upper_bound+1):
        i = upper_bound +l
        if n_MB[j,i]>nmaxr:
            nmaxr=n_MB[j,i]
            imaxr=l
    dAdx=np.zeros(pm.jmax,dtype='complex')

    # Extraplorate the Hxc vector potential for the low density regions
    for i in range(imaxl+1):
        l=imaxl-i
        if n_MB[j,l]<1e-8:
            dAdx[:]=np.gradient(A_KS[j,:],pm.deltax)
            A_KS[j,l]=8*A_KS[j,l+1]-8*A_KS[j,l+3]+A_KS[j,l+4]+dAdx[l+2]*12.0*pm.deltax 
    for i in range(upper_bound +1-imaxr):
        l=(upper_bound+imaxr+i)
        if n_MB[j,l]<1e-8:
            dAdx[:]=np.gradient(A_KS[j,:],pm.deltax)
            A_KS[j,l]=8*A_KS[j,l-1]-8*A_KS[j,l-3]+A_KS[j,l-4]-dAdx[l-2]*12.0*pm.deltax
    return A_KS

# Function to filter out 'noise' occuring between calculation of the exact TDSE solution and the present KS solution
def Filter(A_KS,j,exp):
    A_Kspace=np.zeros(pm.jmax,dtype='complex')
    A_Kspace=momentumspace(A_KS[j,:])
    A_Kspace[:]*=exp[:]
    A_KS[j,:]=realspace(A_Kspace).real
    return A_KS

# Function to solve TDKSEs using the Crank-Nicolson method
def SolveKSE(V_KS,A_KS,Psi,j,frac1,frac2,z):
    Mat=sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')					 						
    for i in range(pm.jmax):
        Mat[i,i]=1.0+0.5j*pm.deltat*(1.0/pm.deltax**2+0.5*A_KS[j,i]**2+V_KS[j,i])
    for i in range(pm.jmax-1):
        Mat[i,i+1]=-0.5j*pm.deltat*(0.5/pm.deltax-(frac1)*1.0j*A_KS[j,i+1]-(frac1)*1.0j*A_KS[j,i])/pm.deltax
    for i in range(1,pm.jmax):
        Mat[i,i-1]=-0.5j*pm.deltat*(0.5/pm.deltax+(frac1)*1.0j*A_KS[j,i-1]+(frac1)*1.0j*A_KS[j,i])/pm.deltax
    for i in range(pm.jmax-2):	
        Mat[i,i+2]=-0.5j*pm.deltat*(1.0j*A_KS[j,i+2]+1.0j*A_KS[j,i])*(frac2)/pm.deltax
    for i in range(2,pm.jmax):
        Mat[i,i-2]=0.5j*pm.deltat*(1.0j*A_KS[j,i-2]+1.0j*A_KS[j,i])*(frac2)/pm.deltax

    # Solve the TDKS equations 
    Mat=Mat.tocsr()
    Matin=-(Mat-sparse.identity(pm.jmax,dtype='complex'))+sparse.identity(pm.jmax,dtype='complex')
    for i in range(pm.NE):
        B=Matin*Psi[i,z,:]
        z=z*(-1)+1 # Only save two times at any point
        Psi[i,z,:]=spla.spsolve(Mat,B)
        z=z*(-1)+1	
    return V_KS,A_KS,Psi,z

# Function to calculate the current density
def CalculateCurrentDensity(n,n_MB,upper_bound,j):
    J=RE_Utilities.continuity_eqn(pm.jmax,pm.deltax,pm.deltat,n[j,:],n[j-1,:])
    if pm.im==1:
        for j in range(pm.jmax):
            for k in range(j+1):
                x=k*pm.deltax-pm.xmax
                J[j]-=abs(pm.im_petrb(x))*n[j,k]*pm.deltax
    #J=ExtrapolateCD(J,j,n,n_MB,upper_bound)
    return J

# Function to calculate the KS vector (and finally scalar) potential
def CalculateKS(V_KS,A_KS,J_KS,Psi,j,upper_bound,frac1,frac2,z,tol,n_MB,J_MB,cost_n,cost_J,A_min,n_KS,Apot,exp):

    # Set initial trial vector potential as previous time-step's vector potential
    A_KS[j,:]=A_KS[j-1,:] 
    V_KS,A_KS,Psi,z=SolveKSE(V_KS,A_KS,Psi,j,frac1,frac2,z)

    # Calculate KS charge density
    n_KS[j,:]=0
    z=z*(-1)+1 # Only save two times at any point
    for i in range(pm.NE):
        n_KS[j,:]+=abs(Psi[i,z,:])**2
    z=z*(-1)+1
    J_KS[j,:]=CalculateCurrentDensity(n_KS,n_MB,upper_bound,j) # Calculate KS current density
    J_MB[j,:]=CalculateCurrentDensity(n_MB,n_MB,upper_bound,j) # Calculate MB current density

    # Evaluate cost functions corresponding to present vector potential
    cost_J[j]=sum(abs(J_KS[j,:]-J_MB[j,:]))*pm.deltax 
    cost_n[j]=sum(abs(n_KS[j,:]-n_MB[j,:]))*pm.deltax

    # Set initial trial vector potential as reference vector potential
    A_min[:]=A_KS[j,:] 
    cost_min=2.0
    count=1
    count_max=100 # Needs to be large, as the algorithm has to converge
    mix=1.0 # Mixing parameter for RE

    while (count<=count_max): # Exit condition: KS and exact current density are equal at all points
        cost_old=cost_J[j] # Keep track of convergence
        if count%10==0:
            mix*=0.5
            if tol<1e-3: # Minimum accuracy limit
                tol*=10 # Increase allowed convergence tolerance for J_check
                A_KS[j,:]=A_KS[j-1,:] # Reset vector potential
                V_KS,A_KS,Psi,z=SolveKSE(V_KS,A_KS,Psi,j,frac1,frac2,z) # Solve Schrodinger equation for KS system using initial trial potential
                n_KS[j,:]=0
                z=z*(-1)+1 # Only save two times at any point
                for i in range(pm.NE):
                    n_KS[j,:]+=abs(Psi[i,z,:])**2
                z=z*(-1)+1
                J_KS[j,:]=CalculateCurrentDensity(n_KS,n_MB,upper_bound,j)
                cost_J[j]=sum(abs(J_KS[j,:]-J_MB[j,:]))*pm.deltax
                cost_n[j]=sum(abs(n_KS[j,:]-n_MB[j,:]))*pm.deltax
            else:
                mix=1.0
                A_KS[j,:]=A_min[:]
                break
        A_KS[j,:]+=mix*(J_KS[j,:]-J_MB[j,:])/n_MB[j,:] # Update vector potential
        A_KS=ExtrapolateVectorPotential(A_KS,n_MB,j,upper_bound) # Extrapolate vector potential from low density regions to edges of system
        A_KS=Filter(A_KS,j,exp) # Remove high frequencies from vector potential
        V_KS,A_KS,Psi,z=SolveKSE(V_KS,A_KS,Psi,j,frac1,frac2,z) # Solve KS equations using updated vector potential
        n_KS[j,:]=0
        z=z*(-1)+1 # Only save two times at any point
        for i in range(pm.NE):
            n_KS[j,:]+=abs(Psi[i,z,:])**2
        z=z*(-1)+1
        J_KS[j,:]=CalculateCurrentDensity(n_KS,n_MB,upper_bound,j) # Calculate updated KS current density
        cost_J[j]=sum(abs(J_KS[j,:]-J_MB[j,:]))*pm.deltax 
        cost_n[j]=sum(abs(n_KS[j,:]-n_MB[j,:]))*pm.deltax   
        if cost_J[j]<cost_min:  # Keep present vector potential for reference if produces lower cost function evaluation
            cost_min=cost_J[j]
            A_min[:]=A_KS[j,:]
        J_check=RE_Utilities.compare(pm.jmax,J_KS[j,:],J_MB[j,:],tol) # Check if KS and exact current density are equal
        if J_check:
            A_KS[j,:]=A_min[:] # Go with the best answer
            z=z*(-1)+1 # Only save two times at any point
            break
        count+=1
        if count>=count_max:
            A_KS[j,:]=A_min[:] # Go with the best answer
            z=z*(-1)+1 # Only save two times at any point
            break
    string='REV: t = ' + str(j*pm.deltat) + ', tol = ' + str(tol) + ', current error = ' + str(cost_J[j]) + ', density error = ' + str(cost_n[j])
    sprint.sprint(string,1,1,pm.msglvl)
    Apot[:]=0 # Change guage so only have scalar potential
    for i in range(pm.jmax): # Calculate full KS scalar potential
        for k in range(i+1):
            Apot[i]+=((A_KS[j,k]-A_KS[j-1,k])/pm.deltat)*pm.deltax
    V_KS[j,:]+=Apot[:]
    V_KS[j,:]+=V_KS[0,(pm.jmax-1)*0.5]-V_KS[j,(pm.jmax-1)*0.5]
    return n_KS,V_KS,J_KS,Apot,z

# Main control function
def main(approx):

    # Constants used in the code
    sqdx=math.sqrt(pm.deltax) 												
    upper_bound = int((pm.jmax-1)/2.0)								
    imax=pm.imax+1
    if pm.TD==0:
        imax=1
    mu=1.0 # Mixing for the ground-state KS algorithm
    z=0
    alpha=1 # Strength of noise control
    frac1=1.0/3.0
    frac2=1.0/24.0

    # Initalise matrices
    T=np.zeros((2,pm.jmax),dtype='complex')
    T[0,:]=np.ones(pm.jmax,dtype='complex')/pm.deltax**2									
    T[1,:]=-0.5*np.ones(pm.jmax,dtype='float')/pm.deltax**2									
    n_MB=np.zeros((imax,pm.jmax),dtype='float')									
    n_KS=np.zeros((imax,pm.jmax),dtype='float')
    J_KS=np.zeros((imax,pm.jmax),dtype='float')									
    J_MB=np.zeros((imax,pm.jmax),dtype='float')
    cost_n=np.zeros(imax,dtype='float')								
    cost_J=np.zeros(imax,dtype='float')
    exp=np.zeros(pm.jmax,dtype='float')								
    CNRHS=np.zeros(pm.jmax, dtype='complex')					
    CNLHS=sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')					
    Mat=sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')					
    Matin=sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')				
    V_ext=np.zeros(pm.jmax,dtype='complex')
    V_KS=np.zeros((imax,pm.jmax),dtype='complex')
    V_h=np.zeros((imax,pm.jmax),dtype='float')									
    V_xc=np.zeros((imax,pm.jmax),dtype='complex')
    V_Hxc=np.zeros((imax,pm.jmax),dtype='complex')								
    A_KS=np.zeros((imax,pm.jmax),dtype='complex')									
    A_min=np.zeros(pm.jmax,dtype='complex')									
    Apot=np.zeros(pm.jmax,dtype='complex')
    U_KS=np.zeros((imax,pm.jmax),dtype='float')
    U_MB=np.zeros((imax,pm.jmax),dtype='float')
    petrb=np.zeros(pm.jmax,dtype='complex')

    # Begin
    n_MB=ReadInput(approx,n_MB,0,imax) # Read in exact charge density obtained from code
    V_KS,n_KS,Psi,V_ext=GroundState(V_KS,n_MB,mu,sqdx,V_ext,T,n_KS) # Calculate (or, if already obtained, check) ground-state KS potential
    V_h[0,:]=Hartree(n_KS[0,:]) # Calculate the Hartree potential
    V_Hxc[0,:]=V_KS[0,:]-V_ext[:] # Calculate the Hartree exhange-correlation potential
    V_xc[0,:]=V_Hxc[0,:]-V_h[0,:] # Calculate the exchange-correlation potential
    f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_vks.db', 'w') # KS potential	
    pickle.dump(V_KS[0,:].real,f)				
    f.close()
    f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_vh.db', 'w') # H potential	
    pickle.dump(V_h[0,:],f)				
    f.close()
    f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_vxc.db', 'w') # XC potential	
    pickle.dump(V_xc[0,:].real,f)				
    f.close()
    if pm.TD==1:

        # Time-dependence
        n_MB=ReadInput(approx,n_MB,1,imax) # Read in exact charge density obtained from code
        for i in range(pm.jmax):
            petrb[i]=pm.petrb((i*pm.deltax-pm.xmax))
            exp[i]=math.exp(-alpha*(i*pm.deltax-pm.xmax)**2)
        V_KS[:,:]=V_KS[0,:]+petrb[:] # Add the perturbing field to the external potential and the KS potential
        V_KS[0,:]-=petrb[:]
        tol=1e-13 # Set inital convergence tolerance for exact and KS current densities
        try:
            counter = 0
            for j in range(1,imax): # Propagate from the ground-state
                n_KS,V_KS,J_KS,Apot,z=CalculateKS(V_KS,A_KS,J_KS,Psi,j,upper_bound,frac1,frac2,z,tol,n_MB,J_MB,cost_n,cost_J,A_min,n_KS,Apot,exp) # Calculate KS potential
                U_KS[j,:]=J_KS[j,:]/n_KS[j,:] # Calculate KS velocity field
                V_h[j,:]=Hartree(n_KS[j,:])
                V_Hxc[j,:]=V_KS[j,:]-(V_ext[:]+petrb[:])
                V_xc[j,:]=V_KS[j,:]-(V_ext[:]+petrb[:]+V_h[j,:])
                counter += 1
        except:
            print
            print 'REV: Stopped at timestep ' + str(counter) + '! Outputing all quantities'
        file_name=open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_vks.db', 'w') # KS potential	
        pickle.dump(V_KS[:,:].real,file_name)				
        file_name.close()
        file_name=open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_vh.db', 'w') # H potential	
        pickle.dump(V_h[:,:],file_name)				
        file_name.close()
        file_name=open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_vxc.db', 'w') # xc potential	
        pickle.dump(V_xc[:,:].real,file_name)				
        file_name.close()


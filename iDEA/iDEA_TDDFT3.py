#--------------------------------------------------------------------------------------

from math import *									#Import all necessary librarys.
from numpy import *
from scipy.linalg import eig_banded, solve
from scipy import linalg as la
from scipy import special
from scipy import sparse
from scipy.sparse import linalg as spla
import parameters as pm
import pickle
import sys
import os as os

#--------------------------------------------------------------------------------------

print
print 'To calculate the time-dependent Kohn-Sham potential enter 1.'			#User input.
print 'For just the ground-state Kohn-Sham potential enter 0.'
TD = raw_input('(TD=1/GS=0):')
if int(TD) != 1 and int(TD) != 0:
		sys.exit("Input error. Please, it's not rocket science.")
print 
print 'Do you know the ground-state Kohn-Sham potential?'				#Choice of DFT or TDDFT output.
selection = raw_input('(Yes=1/No=0):')
if int(selection) != 1 and int(selection) != 0:
	sys.exit("Input error. Please, it's not rocket science.")

#---------------------------------------Parameters-------------------------------------

print
print 'Importing variables from parameters file.'

Nx = pm.jmax										#Size of spatial array.
NT = pm.imax										#Number of steps through time, starts at 												t=0.
L = 2*pm.xmax										#Length of the system.
Tmax = pm.tmax										#Total time the system evolves over.
dx = L/(Nx-1)			 							#Spatial step.
dt = Tmax/(NT-1)									#Time step.
sqdx = sqrt(dx)
qmax = 0.5										#Maximum value for mixing parameter.
qmin = 0.1										#Minimum value for mixing parameter.
mmax = 100										#Maximum number of iterations.
mu = 1.0
c = pm.acon										#Softening coefficent for the 1D Coulomb 												repultion.
if int(TD) == 1:
	Nt = NT										#If I want to run the code for less time 												than the full system length.
else:
	Nt = 1										#Only require the GSKS.

#----------------------------------------Matrices---------------------------------------

T = zeros((2,Nx),dtype='complex')							#Kinetic energy matrix operator, diagonal 												and off-diagonal.
T[0,:] = ones(Nx)/dx**2									#Diagonal.
T[1,:] = -0.5*ones(Nx)/dx**2								#Off-diagonal, 2nd order approximation.
n_x0 = zeros((Nt,Nx))									#Matrix for the exact charge denisty.
n_x = zeros((Nt,Nx))									#Matrix for the charge density of the 												electrons.
Psi0 = zeros((Nt,Nx), dtype='complex')							#Wave-function for each particle, evolve 												through time.
Psi1 = zeros((Nt,Nx), dtype='complex')
Psi2 = zeros((Nt,Nx), dtype='complex')
J_x = zeros((Nt,Nx))									#Matrix for the current density.
J_x0 = zeros((Nt,Nx))									#Matrix for the current density.
V_h = zeros((Nt,Nx))									#Matrix for the Hartree potential.
V_xc = zeros((Nt,Nx))									#Matrix for the exchange correlation 												potential.
cost_J = zeros(Nt)									#Array for the cost value.
CNRHS = zeros(Nx, dtype='complex')							#Array for the right hand side of Crank 											Nicolson method.
CNLHS = sparse.lil_matrix((Nx,Nx),dtype='complex')					#Matrix for the left hand side of the Crank 												Nicolson method.
Mat = sparse.lil_matrix((Nx,Nx),dtype='complex')					#Matrix for the left hand side of the Crank 												Nicolson method.
Matin = sparse.lil_matrix((Nx,Nx),dtype='complex')					#Inverted Matrix for the right hand side of 												the Crank Nicolson method.
V = zeros((Nt,Nx))									#Matrix for the Kohn-Sham potential.
A = zeros((Nt,Nx))									#Phase correction matrix.
A_min = zeros((Nt,Nx))									#Used to keep track of the best A value.
Apot = zeros(Nx)
V_xc_H = zeros((Nt,Nx))									#Hartree and XC poetntial.
E = zeros((Nt,Nx))
E_xc = zeros((Nt,Nx))
cost_n = zeros(Nt)
U_F = zeros((Nt,Nx))
U_F0 = zeros((Nt,Nx))
Jdx = zeros((Nt,Nx))
ndt = zeros((Nt,Nx))
cost_U = zeros(Nt)
eN = zeros((Nt,Nx))									#Electron number as a function of x for the 												KS.
eN0 = zeros((Nt,Nx))									#Electron number as a function of x for the 												MB.
a_F = zeros((Nt,Nx))
a_F0 = zeros((Nt,Nx))
dApotdx = zeros(Nx)

#---------------------------------------Functions---------------------------------------

def Currentdensity(j, n):								#Calculation of the current density via 											continity equation.
	J = zeros(Nx)
	if j != 0:
		for i in range(Nx):			
			for k in range(i+1):
				J[i] += -dx*(n[j,k]-n[j-1,k])/dt

	#--------------------------------Extrapolator-----------------------------------

	nmaxl = 0									#Extrapolate the boundary velocity field.
	imaxl = 0
	for i in range(int(0.5*(Nx-1))+1):
		if n_x0[j,i]>nmaxl:
			nmaxl = n_x0[j,i]
			imaxl = i
	nmaxr = 0
	imaxr = 0
	for l in range(int(0.5*(Nx-1))+1):
		i = int(0.5*(Nx-1)+l)
		if n_x0[j,i]>nmaxr:
			nmaxr = n_x0[j,i]
			imaxr = l
	U = zeros(Nx)
	U[:] = J[:]/n[j,:]	
	dUdx = zeros(Nx)
	for i in range(imaxl+1):
		l = imaxl-i
		if n_x0[j,l] < 1e-6:
			dUdx[:] = gradient(U[:], dx)
			U[l] = 8*U[l+1]-8*U[l+3]+U[l+4]+dUdx[l+2]*12.0*dx
	for i in range(int(0.5*(Nx-1)-imaxr+1)):
		l = int(0.5*(Nx-1)+imaxr+i)
		if n_x0[j,l] < 1e-6:
			dUdx[:] = gradient(U[:], dx)
			U[l] = 8*U[l-1]-8*U[l-3]+U[l-4]-dUdx[l-2]*12.0*dx
	
	#------------------------------------------------------------------------------

	J[:] = n[j,:]*U[:]								#So that the boundary conditions are 												inherent.
	return J[:]

def realspace(vector):									#Define function for Fourier transforming 												into real-space.
	mid_k = int(0.5*(Nx-1))
	fftin = zeros(Nx-1, dtype='complex')
	fftin[0:mid_k+1] = vector[mid_k:Nx]
	fftin[Nx-mid_k:Nx-1] = vector[1:mid_k]
	fftout = fft.ifft(fftin)
	func = zeros(Nx, dtype='complex')
	func[0:Nx-1] = fftout[0:Nx-1]
	func[Nx-1] = func[0]
	return func

def momentumspace(func):								#Define function for Fourier transforming 												into k-space.
	mid_k = int(0.5*(Nx-1))
	fftin = zeros(Nx-1, dtype='complex')
	fftin[0:Nx-1] = func[0:Nx-1] + 0.0j
	fftout = fft.fft(fftin)
	vector = zeros(Nx, dtype='complex')
	vector[mid_k:Nx] = fftout[0:mid_k+1]
	vector[1:mid_k] = fftout[Nx-mid_k:Nx-1]
	vector[0] = vector[Nx-1].conjugate()
	return vector

def Hartree(n):

	n_k = momentumspace(n)*dx/L							#Define function for generating the Hartree 												potential for a given charge density.
	X_x = zeros(Nx)
	for i in range(Nx):
		x = i*dx-0.5*L
		X_x[i] = 1.0/(abs(x)+c)
	X_k = momentumspace(X_x)*dx/L
	V_k = zeros(Nx, dtype='complex')
	V_k[:] = X_k[:]*n_k[:]
	fftout = realspace(V_k).real*L/dx
	V_hx = zeros(Nx)
	V_hx[0:0.5*(Nx+1)] = fftout[0.5*(Nx-1):Nx]
	V_hx[0.5*(Nx+1):Nx-1] = fftout[1:0.5*(Nx-1)]
	V_hx[Nx-1] = V_hx[0]
	V_hx = sqrt(Nx)*V_hx
	return V_hx

def Potential(j, i, V):									#Potential generator.

	V[j,i] = V[0,i]
	x = i*dx-0.5*L
	if j > 0: 
		V[j,i] += pm.petrb(x)							#Electric field applied at t=0.
	return V[j,i]

#-----------------------------------Reverse-engineering---------------------------------

def KohnSham(V, j, Psi0, Psi1, Psi2, n_x0, J_x0, A, mmax, qmin, cost_U, n_x):		#Kohn-Sham potential for t>0.

	#----------------------------------Boundary-------------------------------------#Allows the edges of the system to decay to 												zero.

	nmaxl = 0
	imaxl = 0
	for i in range(int(0.5*(Nx-1))+1):
		if n_x0[j,i]>nmaxl:
			nmaxl = n_x0[j,i]
			imaxl = i
	nmaxr = 0
	imaxr = 0
	for l in range(int(0.5*(Nx-1))+1):
		i = int(0.5*(Nx-1)+l)
		if n_x0[j,i]>nmaxr:
			nmaxr = n_x0[j,i]
			imaxr = l

	#-------------------------------------------------------------------------------
	
	A[j,:] = A[j-1,:]
	n_x, V, Psi0, Psi1, Psi2, J_x, cost_J[j], cost_n[j] = CrankNicolson(V, n_x0, Psi0\
	, Psi1, Psi2, j, J_x0, A, n_x)

	#-------------------------Initial Velocity field--------------------------------

	U_F0[j,:] = J_x0[j,:]/n_x0[j,:]							#Calculate the velocity field.

	#-------------------------------------------------------------------------------

	q = qmax
	m = 0
	cost_min = 3.0									#Makes sure that the first cost is recorded.
	while cost_n[j] > 0:								#Uses a direct reverse-engineering method.
		
		cost_old = cost_n[j]
		m += 1
		for i in range(Nx):
			A[j,i] += q*(J_x[j,i]-J_x0[j,i])/n_x0[j,i]

		#--------------------------------Extrapolator----------------------------

		dAdx = zeros(Nx)							#Extrapolate the vector potential at the 												boundaries.
		for i in range(imaxl+1):
			l = imaxl-i
			if n_x0[j,l] < 1e-3:
				dAdx[:] = gradient(A[j,:], dx)
				A[j,l] = 8*A[j,l+1]-8*A[j,l+3]+A[j,l+4]+dAdx[l+2]*12.0*dx
		for i in range(int(0.5*(Nx-1)-imaxr+1)):
			l = int(0.5*(Nx-1)+imaxr+i)
			if n_x0[j,l] < 1e-3:
				dAdx[:] = gradient(A[j,:], dx)
				A[j,l] = 8*A[j,l-1]-8*A[j,l-3]+A[j,l-4]-dAdx[l-2]*12.0*dx

		#------------------------------Smooth function--------------------------

		A_Kspace = zeros(Nx, dtype='complex')					#Get rid of noise using Fourier transporm.
		A_Kspace = momentumspace(A[j,:])
		alpha = 0.5

		for i in range(Nx):
			k = i*dx-0.5*L
			A_Kspace[i] = A_Kspace[i]*exp(-alpha*k**2)			#Envelope functions to eliminate high 												frequency components (noise).
		A[j,:] = realspace(A_Kspace).real	

		n_x, V, Psi0, Psi1, Psi2, J_x, cost_J[j], cost_n[j] = CrankNicolson(V, n_x0,\
		 Psi0, Psi1, Psi2, j, J_x0, A, n_x)

		#-----------------------------------------------------------------------

		afile.write('%s %s\n' % (m,cost_n[j]))					#Keep track of the charge density cost.
		cfile.write('%s %s\n' % (m,cost_J[j]))					#Keeps track of the current density cost.

		if cost_n[j] <= cost_min:
			cost_min = cost_n[j]
			A_min[j,:] = A[j,:]
		if m == mmax:								#Max number of iterations. 
			print 'Does not converge in %s iterations.' % mmax
			print 'Convergence parameter halfed...' 
			q = 0.5*q
			A[j,:] = 0
			m = 0
			n_x, V, Psi0, Psi1, Psi2, J_x, cost_J[j], cost_n[j] = \
			CrankNicolson(V, n_x0, Psi0, Psi1, Psi2, j, J_x0, A, n_x)
			if q <= qmin:
				print 'Convergence parameter has reached is minium.' 
				print 'Reverting back to potential corresponding to the lowest overall cost.'
				A[j,:] = A_min[j,:]
				break
		if abs(cost_old-cost_n[j]) < 1e-12:					#Stops if there is no noticable change to 												the system.
			if cost_n[j] == cost_min:
				print 'System has converged after %s iterations.' % m
				break
			else:
				print 'System has converged after %s iterations.' % m
				print 'WARNING: converged cost is NOT the overall minimum cost!'
				print 'Reverting back to potential corresponding to the lowest cost.'
				A[j,:] = A_min[j,:]
				break

	f = open('AKspace(t=%s).out' % j, 'w')						#The Fourier transform of A.
	for i in range(Nx):
		x = i*dx-0.5*L
		f.write('%s %s\n' % (x, A_Kspace[i].real))
	f.close()

	n_x, V, Psi0, Psi1, Psi2, J_x, cost_J[j], cost_n[j] = CrankNicolson(V, n_x0\
	, Psi0, Psi1, Psi2, j, J_x0, A, n_x)
	return n_x, V, cost_J[j], Psi0, Psi1, Psi2, J_x, A, cost_n[j], cost_U[j]	#Returns the charge density and the 												Kohn-Sham potential.

#--------------------------------------(t=0) TISE---------------------------------------

def Groundstate(V, n_x0, j, retflag, mu, setup):					#Uses TISE to find initial ground state 											wave-functions.
	HGS = copy(T)									#Reset the Hamiltonian.
	if setup == 0:
		for i in range(Nx):
			if abs(n_x[j,i]-n_x0[j,i]) > noise:
				V[j,i] += mu*(n_x[j,i]**(1.0/20.0)-\
				n_x0[j,i]**(1.0/20.0))

	HGS[0,:] += V[j,:]								#Add the Kohn-Sham potential to the 												Hamiltonian.
	K, U = eig_banded(HGS, True)							#Find initial wave-functions and 												eigen-energies.
	Psi0[j,:] = U[:,0]/sqdx								#Normalise the wave-functions.
	Psi1[j,:] = U[:,1]/sqdx
	Psi2[j,:] = U[:,2]/sqdx

	n_x[j,:] = abs(Psi0[j,:])**2+abs(Psi1[j,:])**2+abs(Psi2[j,:])**2		#Charge density composed from the ground 												state and 1st excited state wave-functions.
	cost_n[j] = sum(abs(n_x0[j,:]-n_x[j,:]))*dx

	if retflag == 0:
		return cost_n[j]							#The cost the optimiser recieves.
	if retflag == 1:
		return n_x, Psi0, Psi1, Psi2, V, cost_n[j]				#Cost function matches the standard 												definition.

#--------------------------Crank Nicolson method evolves TDSE------------------------- 

def CrankNicolson(V, n_x0, Psi0, Psi1, Psi2, j, J_x0, A, n_x):
	
	Mat, V = LHS(V, j, A)								#The Hamiltonian here is using the 												Kohn-Sham potential.
	Mat = Mat.tocsr()
	Matin = -(Mat-sparse.identity(Nx, dtype=cfloat))+\
		sparse.identity(Nx, dtype=cfloat)
	B0 = Matin*Psi0[j-1,:]
	Psi0[j,:] = spla.spsolve(Mat, B0)						#Solve the Crank Nicolson equation to get 												the wave-function at dt later.	
	B1 = Matin*Psi1[j-1,:]
	Psi1[j,:] = spla.spsolve(Mat, B1)						#Solve the Crank Nicolson equation to get 												the wave-function at dt later.
	B2 = Matin*Psi2[j-1,:]
	Psi2[j,:] = spla.spsolve(Mat, B2)						#Solve the Crank Nicolson equation to get 												the wave-function at dt later.
	n_x[j,:] = abs(Psi0[j,:])**2+abs(Psi1[j,:])**2+abs(Psi2[j,:])**2
	J_x[j,:] = Currentdensity(j, n_x)
	cost_J[j] = sum(abs(J_x0[j,:]-J_x[j,:]))*dx
	cost_n[j] = sum(abs(n_x0[j,:]-n_x[j,:]))*dx

	return n_x, V, Psi0, Psi1, Psi2, J_x, cost_J[j], cost_n[j]

#---------------------------------------------------------------------------------------

def LHS(V, j, A):									#Left hand side of the Crank Nicolson 												method.

	for i in range(Nx):
		V[j,i] = Potential(j, i, V)
		CNLHS[i,i] = 1.0+0.5j*dt*(1.0/dx**2+0.5*A[j,i]**2+V[j,i])
		if i < Nx-1:
			CNLHS[i,i+1] = -0.5j*dt*(0.5/dx-(1.0/3.0)*1.0j*A[j,i+1]-\
			(1.0/3.0)*1.0j*A[j,i])/dx
		if i > 0:
			CNLHS[i,i-1] = -0.5j*dt*(0.5/dx+(1.0/3.0)*1.0j*A[j,i-1]+\
			(1.0/3.0)*1.0j*A[j,i])/dx
		if i < Nx-2:
			CNLHS[i,i+2] = -0.5j*dt*(1.0j*A[j,i+2]+1.0j*A[j,i])\
			*(1.0/24.0)/dx
		if i > 1:
			CNLHS[i,i-2] = 0.5j*dt*(1.0j*A[j,i-2]+1.0j*A[j,i])\
			*(1.0/24.0)/dx
	return CNLHS, V

#-------------------------------------3D inputs----------------------------------------

dendir = os.path.join("CDensity")
origdir = os.getcwd()
os.chdir(dendir)  
if int(TD) == 1:      
	print
	print 'Reading input...'
	ProbPsi = open("ProbPsi(Nx=%s,Nt=%s).db" % (Nx,NT), "r")			#Exact solution to the three electron 												system.
	n_x0[:,:] = pickle.load(ProbPsi)[:,:]						#Defines n_x0 from the exact wave-function.
	ProbPsi.close()  
else:    
	print
	print 'Reading input...'
	ProbPsi = open("ProbPsi(Nx=%s,Nt=%s).db" % (Nx,NT), "r")			#Exact solution to the three electron 												system.
	n_x0[0,:] = pickle.load(ProbPsi)[:]						#Defines n_x0 from the exact wave-function.
	ProbPsi.close()
os.chdir(origdir)
print ' Charge-density loaded.'	

print
print 'Generating run information file...'
Tsim = dt*(Nt-1)
f = open('RunInfo.out', 'w')								#Gives relavant run information.
f.write('Number of spatial grid points %s\n' % Nx)
f.write('Number of time grid points %s\n' % Nt)
f.write('Length of the system %s\n' % L)
f.write('Time length of the system %s\n' % Tsim)
f.write('dx=%s\n' % dx)
f.write('dt=%s\n' % dt)
f.write('Maximum mixing parameter %s\n' % qmax)
f.write('Minimum mixing parameter %s' % qmin)
f.close()

#--------------------Evolve through time (Crank Nicolson method)------------------------

efile = open('Physicalerrors.out', 'w')							#Records the physics to insure that the 											system stays physical.
pfile = open('costcurrent.out', 'w')							#Plot to show the cost at each time step.
bfile = open('costcharge.out', 'w')

for j in range(Nt):

	if j == 0:
		if int(selection) == 1:							#If the ground-state potential is known. 
			print
			print 'Loading ground-state Kohn-Sham potential....'
			noise = 0
			GSP = open("Ground-State.db", "r")
			V[j,:] = pickle.load(GSP)[:]
			GSP.close()
			n_x, Psi0, Psi1, Psi2, V, cost_n[j] = Groundstate(V, n_x0, j, 1, 0, 1)
			print '--------------------------------------------------------------'
			print 'Ground-state cost = %s' % cost_n[j]
			print '--------------------------------------------------------------'
			if cost_n[j] > 1e-5:
				print 'Error: your ground-state Kohn-Sham potential is not accurate enough.'
				selection = 0

		if int(selection) == 0:							#If the ground-state potential is not 												known. 
			print
			print 'Calculating ground-state Kohn-Sham potential....'

			#----------------------------------Noise------------------------

			#print
			#print 'Calculating charge-density noise magnitude...'		#Calculates the symmetry noise to stop 												symmetry breaking
			#noise_max = 0
			#for i in range(Nx):
			#	noise = abs(n_x[j,i]-n_x[j,Nx-1-i]-(n_x0[j,i]\
			#	-n_x0[j,Nx-1-i]))
			#	if noise > noise_max:
			#		noise_max = noise
			#	else:
			#		noise = noise_max
			#print 'Symmetrical noise magnitude = %s' % noise
			#if noise > 1e-5:
			#	print 'WARNING: noise is very high!'

			#f = open('Noise.out', 'w')					#Plots the symmetry noise.
			#for i in range(Nx):
			#	f.write('%s %s\n' % (i*dx-0.5*L, n_x[j,i]-\
			#	n_x[j,Nx-1-i]))
			#f.close()
			#for i in range(Nx):
			#	if n_x[j,i]-n_x[j,Nx-1-i] > 1e-5:
			#		sys.exit("CRITICAL ERROR: symmetry broken!")	#Code stops if the system is not symmetric 												enough.
			noise = 1e-14

			#---------------------------------------------------------------

			for i in range(Nx):
				x = i*dx-0.5*L
				V[j,i] = pm.well(x)
			n_x, Psi0, Psi1, Psi2, V, cost_n[j] = Groundstate(V, n_x0, j,\
			 1, mu, 1)							#First guess is an LDA like potential.
			f = open('Vext(t=%s).out' % j, 'w')
			for i in range(Nx):
				x = i*dx-0.5*L
				f.write('%s %s\n' % (x, pm.well(x)))
			f.close() 
			f = open('Den[0](t=%s).out' % j, 'w')
			for i in range(Nx):
				f.write('%s %s\n' % (i*dx-0.5*L, n_x0[j,i]))
			f.close()
			f = open('Den(t=%s).out' % j, 'w')
			for i in range(Nx):
				f.write('%s %s\n' % (i*dx-0.5*L, n_x[j,i]))
			f.close()
			print
			print '--------------------------------------------------------------'
			print 'Inital guess cost = %s' % cost_n[j]			#Final cost for that step, has the form of 												the standard deffinition.
			print '--------------------------------------------------------------'
			print 
			print 'Converging local corrections... '			#Finds the GS KS potential.
			while cost_n[j] > noise:
				cost_old = cost_n[j]
				n_x, Psi0, Psi1, Psi2, V, cost_n[j] = Groundstate(V, n_x0, j\
				, 1, mu, 0)
				if abs(cost_n[j]-cost_old) < 1e-15 or cost_n[j] > \
				cost_old:
					mu = mu*0.5
				if mu < 1e-15:
					break
			print '--------------------------------------------------------------'
			print 'Final ground-state cost = %s' % cost_n[j]		#Final cost for that step, has the form of 												the standard deffinition.
			print '--------------------------------------------------------------'
			print 'Saving ground-state Kohn-Sham to file.'
			File = open("Ground-State.db", "w")				#Writes the ground-state potential to file.
			pickle.dump(V[0,:],File)
			File.close()
	else:
		t = j*dt
		print
		print 'Calculating current density from continuity equation...'
		J_x0[j,:] = Currentdensity(j, n_x0)
		print 'Calculating Kohn-Sham potential at t = %s....' % t
		cfile = open('costREC(t=%s).out' % j, 'w')
		afile = open('costREN(t=%s).out' % j, 'w')
		print
		print 'Applying vector correction...'
		n_x, V, cost_J[j], Psi0, Psi1, Psi2, J_x, A, cost_n[j], cost_U[j]= \
		KohnSham(V, j, Psi0, Psi1, Psi2, n_x0, J_x0, A, mmax, qmin, cost_U, n_x)
		Apot[:] = 0
		for i in range(Nx):
			for k in range(i+1):
				Apot[i] += ((A[j,k]-A[j-1,k])/dt)*dx			#Changing guage so only have scalar 												potential.

		cost_n[j] = sum(abs(n_x0[j,:]-n_x[j,:]))*dx
		print '--------------------------------------------------------------'
		print 'Current density cost = %s' % cost_J[j]
		print 'Charge density cost = %s' % cost_n[j]
		print '--------------------------------------------------------------'
		cfile.close()
		afile.close()

		#-------------------------Velocity field------------------------

		U_F[j,:] = J_x[j,:]/n_x[j,:]
		U_F0[j,:] = J_x0[j,:]/n_x0[j,:]

		#-----------------------Acceloration field----------------------

		for i in range(Nx):
			a_F[j,i] = (U_F[j,i]-U_F[j-1,i])/dt
		for i in range(Nx):
			a_F0[j,i] = (U_F0[j,i]-U_F0[j-1,i])/dt

		#---------------------------------------------------------------

	for i in range(Nx):
		V[j,i] = Potential(j, i, V)+Apot[i]
	V[j,:] += V[0,(Nx-1)*0.5]-V[j,(Nx-1)*0.5]					#So that the potentials line up.

#------------------------------------Step function posistion----------------------------

#	dAdx = zeros(Nx)
#	dAdx[:] = gradient(A[j,:], dx)
#	Agradimax = 0
#	dAdxmax = 0
#	for i in range(Nx):
#		if i > int(0.35*Nx) and i < int(0.65*Nx):
#			if dAdx[i] > dAdxmax:
#				Agradimax = i
#				dAdxmax = dAdx[i]
#	print 'Central gradient maximum at x=%s' % (Agradimax*dx-0.5*L)

#-------------------------------------Charge density minimum----------------------------

#	dndxmin = 2.0
#	dndx = zeros(Nx)
#	dndx[:] = gradient(n_x[j,:], dx)
#	for i in range(Nx):
#		if i > int(0.35*Nx) and i < int(0.65*Nx):
#			if abs(dndx[i]) < dndxmin:
#				ngradmin = i
#				dndxmin = abs(dndx[i])
#	print 'Charge density minimum at x=%s' % (ngradmin*dx-0.5*L)

#--------------------------------------Electron number----------------------------------

	for i in range(Nx):
		for k in range(i+1):
			eN0[j,i] += n_x0[j,k]*dx
	eN0[j,:] += -1									#Making it easier to find the integer 												crossover.
	for i in range(Nx):
		for k in range(i+1):
			eN[j,i] += n_x[j,k]*dx
	eN[j,:] += -1

#-----------------------Isolating the exchange correlation potential--------------------

	V_h[j,:] = Hartree(n_x[j,:])							#Calculate the Hartree potential.
	for i in range(Nx):
		x = i*dx-0.5*L
		if j == 0:
			V_xc[j,i] = V[j,i]-(pm.well(x)+V_h[j,i])			#Take away the other potentials leaving 											only the exchange potential.
			V_xc_H[j,i] = V[j,i]-pm.well(x)					#Take away the external potential.
		else:
			V_xc[j,i] = V[j,i]-(pm.well(x)+pm.petrb(x)+V_h[j,i])
			V_xc_H[j,i] = V[j,i]-(pm.well(x)+pm.petrb(x))

#--------------------------Calculating confining electric field-------------------------

	E[j,:] = -gradient(V[j,:], dx)							#The electric filed is minus the gradient 												of the potential.
	E_xc[j,:] = -gradient(V_xc[j,:], dx)						#The electric filed is minus the gradient 												of the potential.
#-----------------------------Output files & error checking-----------------------------
	
	print
	print 'Outputting results...'

	pfile.write('%s %s\n' % (j*dt,cost_J[j]))
	bfile.write('%s %s\n' % (j*dt,cost_n[j]))

	if int(TD) == 1:
		print
		print 'Testing the system for physical errors...'
		Jdx[j,:] = gradient(J_x[j,:], dt)
		ndt[:,i] = gradient(n_x[:,i], dx)
		for i in range(Nx):
			cont = Jdx[j,i]+ndt[j,i]
			if cont > 1e-10:
				efile.write('Continuity at t=%s x=%s: %s\n' % (j*dt,\
				i*dx-0.5*L, cont))					#Check by using continuity equation (should 												equal zero ~10^{-10}).

	efile.write('Normalisation at t=%s: %s\n' % (j*dt, sum(n_x[j,:])*dx))		#Chack the normalisation (should be 2).

	f = open('V_KS(t=%s).out' % j, 'w')						#Kohn-Sham potential.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, V[j,i]))
	f.close()
	f = open('Den(t=%s).out' % j, 'w')						#1D charge density.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, n_x[j,i]))
	f.close()
	f = open('CDen(t=%s).out' % j, 'w')						#1D current density.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, J_x[j,i]))
	f.close()
	f = open('Den[0](t=%s).out' % j, 'w')						#2D charge density.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, n_x0[j,i]))
	f.close()
	f = open('CDen[0](t=%s).out' % j, 'w')						#2D current density.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, J_x0[j,i]))
	f.close()
	f = open('V_xc(t=%s).out' % j, 'w')						#Exchange & correlation potential.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, V_xc[j,i]))
	f.close()
	f = open('V_xc_H(t=%s).out' % j, 'w')						#XC and Hartree potential.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, V_xc_H[j,i]))
	f.close()
	f = open('E(t=%s).out' % j, 'w')						#Electric field of the KS potential.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, E[j,i]))
	f.close()
	f = open('E_xc(t=%s).out' % j, 'w')						#Electric field of the XC potential.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, E_xc[j,i]))
	f.close()
	f = open('U_F(t=%s).out' % j, 'w')						#KS velocity field.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, U_F[j,i]))
	f.close()
	f = open('U_F[0](t=%s).out' % j, 'w')						#Real velocity field.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, U_F0[j,i]))
	f.close()
	f = open('eN(t=%s).out' % j, 'w')						#KS electron number.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, eN[j,i]))
	f.close()
	f = open('eN[0](t=%s).out' % j, 'w')						#Real electron number.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, eN0[j,i]))
	f.close()
	f = open('A(t=%s).out' % j, 'w')						#Time-dependent KS vector potential.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, A[j,i]))
	f.close()
	f = open('a_F(t=%s).out' % j, 'w')						#KS velocity field.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, a_F[j,i]))
	f.close()
	f = open('a_F[0](t=%s).out' % j, 'w')						#Real velocity field.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, a_F0[j,i]))
	f.close()
	f = open('Apot(t=%s).out' % j, 'w')						#Plot the addition to potential to the GSKS.
	for i in range(Nx):
		f.write('%s %s\n' % (i*dx-0.5*L, Apot[i]))
	f.close()
efile.close()
pfile.close()
bfile.close()

if int(TD) == 1:
	print
	print 'Outputting pickle files...'

	File = open("KohnSham.db", "w")							#Writes the generated Kohnsham potnetial to 												a pickle file.
	pickle.dump(V,File)
	File.close()

	File = open("Exchange-correlation.db", "w")					#Writes the generated exchange correlation 												potnetial to a pickle file.
	pickle.dump(V_xc,File)
	File.close()

	File = open("Density.db", "w")							#Writes the 1D density to a pickle file.
	pickle.dump(n_x,File)
	File.close()
	
	File = open("Density[0].db", "w")						#Writes the 3D density to a pickle file.
	pickle.dump(n_x0,File)
	File.close()

	File = open("Currentdensity.db", "w")						#Writes the 1D current density to file.
	pickle.dump(J_x,File)
	File.close()

	File = open("Currentdensity[0].db", "w")					#Writes the 3D current density to file.
	pickle.dump(J_x0,File)
	File.close()

	File = open("Psi0.db", "w")
	pickle.dump(Psi0,File)
	File.close()

	File = open("Psi1.db", "w")
	pickle.dump(Psi1,File)
	File.close()

	print
	print 'Outputs: Ground-state and time-dependent Kohn-Sham potential.' 
	print '         Time-dependent Kohn-Sham charge and current densities.'
	print '         Time-dependent exchange-correlation potential, exchange-correlation and Hartree potential.' 
	print '         velocity field and acceloration field.'
	print '         Physical errors files.'
	print '         Run infomation file.' 
	print '(All potentials have their corresponding electric fields and all time-dependent outputs have a video option.)'
print
print 'Simulation completed successfully.' 
print 'Have a nice day.'

######################################################################################
# Name: Many Body Pertubation Theory                                                 #
######################################################################################
# Author(s): Jack Wetherell                                                          #
######################################################################################
# Description:                                                                       #
# Computes ground-state density of a system using the GW approximation               #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################

# Library imports
import math
import copy
import pickle
import sprint
import numpy as np
import scipy as sp
import parameters as pm
import numpy.linalg as npl
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# Struct to define space-time grid
class SpaceTime:
   def __init__(self):
      self.tau_grid = np.append(np.linspace(0,pm.tau_max,pm.tau_N+1),np.linspace(-pm.tau_max,-float(pm.tau_max)/float(pm.tau_N),pm.tau_N)) # Zero first time ordering
      self.x1_grid = np.linspace(-pm.xmax,pm.xmax,pm.grid)
      self.x2_grid = np.linspace(-pm.xmax,pm.xmax,pm.grid)
      self.tau_N = pm.tau_N # Number of total imaginary time points at EITHER SIDE OF ZERO
      self.tau_N_total = len(self.tau_grid) # Total number of imaginary time points
      self.x_N = len(self.x1_grid)
      self.switch_point = pm.tau_N+1 # Grid point with first negative imaginary time point
      self.dx = float(2*pm.xmax)/float(pm.grid-1)
      self.dtau = float(2*pm.tau_max)/float(self.tau_N_total-1)

# Function to construct the kinetic energy K
def constructK():
   xgrid = np.linspace(-pm.xmax,pm.xmax,pm.grid)
   K = -0.5*sps.diags([1, -2, 1],[-1, 0, 1], shape=(pm.grid,pm.grid), format='csr')/(pm.deltax**2)
   return K

# Function to construct the potential V
def constructV():
   if(pm.starting_orbitals == 'non'):
      xgrid = np.linspace(-pm.xmax,pm.xmax,pm.grid)
      Vdiagonal = []
      for i in range(0,len(xgrid)):
         Vdiagonal.append(pm.well(xgrid[i]))
      V = sps.spdiags(Vdiagonal, 0, pm.grid, pm.grid, format='csr')
   else:
      input_file=open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(pm.starting_orbitals) + '_vks.db','r')
      V = pickle.load(input_file)
   return V

# Function to construct the non-interacting green's function G0 in the time domain
def non_interacting_greens_function(st, occupied, occupied_energies, empty, empty_energies):
   G0 = np.zeros((st.tau_N_total,st.x_N,st.x_N), dtype='complex')
   occupied_tensor = np.zeros((pm.NE,st.x_N,st.x_N), dtype='complex') 
   for n in range(0,pm.NE):
      for i in range(0, st.x_N):
         for j in range(0, st.x_N):
            occupied_tensor[n,i,j] = occupied[i,n]*np.conjugate(occupied[j,n]) # phi_n(x_1)phi_n*(x_2) for occupied states
   empty_tensor = np.zeros((pm.number,st.x_N,st.x_N), dtype='complex') 
   for n in range(0,pm.number):
      for i in range(0, st.x_N):
         for j in range(0, st.x_N):
            empty_tensor[n,i,j] = empty[i,n]*np.conjugate(empty[j,n]) # phi_n(x_1)phi_n*(x_2) for empty states
   for k in range(0,st.tau_N_total):
      tau = st.tau_grid[k]
      string = 'MBPT: computing non-interacting greens function G0, tau = ' + str(tau)
      sprint.sprint(string,1,1,pm.msglvl)
      if(k < st.switch_point): # Construct G0 for positive imaginary time
         for i in range(0,st.x_N):
            for j in range(0,st.x_N):
               G0[k,i,j] = -1.0j*np.sum(empty_tensor[:,i,j] * np.exp(-empty_energies[:]*tau))
      else: # Construct G0 for negative imaginary time
         for i in range(0,st.x_N):
            for j in range(0,st.x_N):
               G0[k,i,j] = 1.0j*np.sum(occupied_tensor[:,i,j] * np.exp(-occupied_energies[:]*tau))
   for i in range(0,st.x_N):
      for j in range(0,st.x_N):
         G0[0,i,j] = 1.0j*np.sum(occupied_tensor[:,i,j]) # Construct G0 for tau=0
   print
   return G0

# Function to construct the alternate definition of the green's function at tau=0
def alternate_greens_function(st, G0, empty, empty_energies):
   Ga0 = np.zeros((st.tau_N_total,st.x_N,st.x_N), dtype='complex')
   Ga0[:,:,:] = G0[:,:,:]
   empty_tensor = np.zeros((pm.number,st.x_N,st.x_N), dtype='complex')
   for i in range(0, st.x_N):
      for j in range(0, st.x_N):
         empty_tensor[:,i,j] = empty[i,:]*np.conjugate(empty[j,:])
   for i in range(0,st.x_N):
      for j in range(0,st.x_N):
         Ga0[0,i,j] = -1.0j*np.sum(empty_tensor[:,i,j])
   return Ga0

# Function to calculate coulomb interaction in the time domain
def coulomb_interaction(st):
   v = np.zeros((st.tau_N_total,st.x_N,st.x_N), dtype='complex')
   for i in range(0,st.x_N):
      for j in range(0,st.x_N):
         v[0,i,j] = (1.0j/(abs(st.x2_grid[j]-st.x1_grid[i])+pm.acon))/(st.dtau) # Softened coulomb interaction
   return v

# Function to calculate the irreducible polarizability P in the time domain
def irreducible_polarizability(st,G,Ga,iteration):
   string = 'MBPT: performing self-consistency (iteration=' + str(iteration+1) + '): P'
   sprint.sprint(string,1,1,pm.msglvl)
   P = np.zeros((st.tau_N_total,st.x_N,st.x_N), dtype='complex')
   for k in range(1,st.tau_N_total):
      for i in range(0,st.x_N):
         for j in range(0,st.x_N):
            P[k,i,j] = -1.0j*G[k,i,j]*G[-k,j,i]
   for i in range(0,st.x_N):
      for j in range(0,st.x_N):
         P[0,i,j] = -1.0j*G[0,i,j]*Ga[0,j,i] # P at tau=0 needs to be construced with the two alternate greens functions 0+ and 0-
   return P

# Function to calculate the screened interaction W_f in the frequency domain
def screened_interaction(st,v_f,P_f,iteration):
   string = 'MBPT: performing self-consistency (iteration=' + str(iteration+1) + '): P, W'
   sprint.sprint(string,1,1,pm.msglvl)
   W_f = np.zeros((st.tau_N_total,st.x_N,st.x_N), dtype='complex')
   W_f_slice = np.zeros((st.x_N,st.x_N), dtype='complex')
   P_f_slice = np.zeros((st.x_N,st.x_N), dtype='complex')
   v_f_slice = np.zeros((st.x_N,st.x_N), dtype='complex') 
   for k in range(0,st.tau_N_total): # At each frequency slice perform W = (I - v P)^-1 v
      v_f_slice[:,:] = v_f[k,:,:]
      P_f_slice[:,:] = P_f[k,:,:]
      W_f_slice = np.dot(npl.inv(np.eye(st.x_N,dtype='complex') - np.dot(v_f_slice,P_f_slice)*st.dx*st.dx),v_f_slice)
      W_f[k,:,:] = W_f_slice[:,:]
   return W_f

# Function to calculate the self energy S in the time domain
def self_energy(st,G,W,iteration):
   string = 'MBPT: performing self-consistency (iteration=' + str(iteration+1) + '): P, W, S'
   sprint.sprint(string,1,1,pm.msglvl)
   S = np.zeros((st.tau_N_total,st.x_N,st.x_N), dtype='complex')
   for k in range(0,st.tau_N_total):
      for i in range(0,st.x_N):
         for j in range(0,st.x_N):
            S[k,i,j] = 1.0j*G[k,i,j]*W[k,i,j] # GW approximation
   return S

# Function to add the hartree potential in the time domain
def add_hartree(st,S,v,density):
   v_slice = np.zeros((st.x_N,st.x_N), dtype='complex')
   v_slice[:,:] = v[0,:,:]
   V_h = np.dot(v_slice,density)*st.dx
   for i in range(0,st.x_N):
      S[0,i,i] = S[0,i,i] + V_h[i]/(st.dx)
   return S

# Function to perfrom the hedin shift in the frequency domain
def hedin_shift(st,S_f,occupied,empty):
   state = np.zeros(st.x_N, dtype='complex')
   S_f_0 = np.zeros((st.x_N,st.x_N), dtype='complex')
   S_f_0[:,:] = S_f[0,:,:]
   state[:] = occupied[:,-1]
   expectation_value1 = np.vdot(state,np.dot(S_f_0,state.transpose()))*st.dx*st.dx # Calculates the expectation value of S(0) in the HOMO state
   state[:] = empty[:,0]
   expectation_value2 = np.vdot(state,np.dot(S_f_0,state.transpose()))*st.dx*st.dx # Calculates the expectation value of S(0) in the LUMO state
   expectation_value = (expectation_value1 + expectation_value2)/2.0 # Calculates the expectation value of S(0) at the fermi energy
   for k in range(0,st.tau_N_total):
      for i in range(0,st.x_N):
         S_f[k,i,i] = S_f[k,i,i] - expectation_value/st.dx
   return S_f

# Function to solve the dyson equation in the frequency domain to update G
def dyson_equation(st,G0_f,S_f,iteration):
   string = 'MBPT: performing self-consistency (iteration=' + str(iteration+1) + '): P, W, S, G'
   sprint.sprint(string,1,1,pm.msglvl)
   G_f = np.zeros((st.tau_N_total,st.x_N,st.x_N), dtype='complex') # Greens function in the frequency domain
   G_f_slice = np.zeros((st.x_N,st.x_N), dtype='complex')
   G0_f_slice = np.zeros((st.x_N,st.x_N), dtype='complex')
   S_f_slice = np.zeros((st.x_N,st.x_N), dtype='complex') 
   for k in range(0,st.tau_N_total): # At each frequency slice perform G = (I - G0 S)^-1 G0
      G0_f_slice[:,:] = G0_f[k,:,:]
      S_f_slice[:,:] = S_f[k,:,:]
      G_f_slice = np.dot(npl.inv(np.eye(st.x_N,dtype='complex') - np.dot(G0_f_slice,S_f_slice)*st.dx*st.dx),G0_f_slice)
      G_f[k,:,:] = G_f_slice[:,:]
   return G_f

# Function to fourier transform a given quantity
def fourier(st, A, inverse):
   if(inverse == 0):
      return -1.0j*np.fft.fft(A,axis=0)*st.dtau
   if(inverse == 1):
      return 1.0j*np.fft.ifft(A,axis=0)/st.dtau

# Function to extract the ground-state density from G
def extract_density(st,G):
   density = np.zeros(st.x_N, dtype='float')
   for i in range(0,st.x_N):
      density[i] = (-1.0j*G[0][i][i]).real
   return density

# Function to test for convergence
def has_converged(density_new, density_old, iteration):
   convergence = abs(npl.norm(density_new-density_old))
   string = 'MBPT: performing self-consistency (iteration=' + str(iteration+1) + '): P, W, S, G. convergence = ' + str(convergence)
   sprint.sprint(string,1,1,pm.msglvl)
   if(convergence < pm.tollerance):
      return True
   else:
      return False

# Function to save all hedin quantities to pickle files
def output_quantities(G0,P,W,S,G):
   output_file = open('outputs/' + str(pm.run_name) + '/data/g0.db','w')
   pickle.dump(G0,output_file)
   output_file.close()
   output_file = open('outputs/' + str(pm.run_name) + '/data/p.db','w')
   pickle.dump(P,output_file)
   output_file.close()
   output_file = open('outputs/' + str(pm.run_name) + '/data/w.db','w')
   pickle.dump(W,output_file)
   output_file.close()
   output_file = open('outputs/' + str(pm.run_name) + '/data/s.db','w')
   pickle.dump(S,output_file)
   output_file.close()
   output_file = open('outputs/' + str(pm.run_name) + '/data/g.db','w')
   pickle.dump(G,output_file)
   output_file.close()

# Main function
def main():

   # Construct the kinetic energy
   K = constructK()

   # Construct the potential
   V = constructV()

   # Constuct the hamiltonian
   H = K + V

   # Compute all wavefunctions
   print 'MBPT: computing eigenstates of single particle hamiltonian'
   solution = spla.eigsh(H, k=pm.grid-2, which='SA', maxiter=1000000)
   energies = solution[0] 
   wavefunctions = solution[1]

   # Normalise all wavefunctions
   length = len(wavefunctions[0,:])
   for i in range(0,length):
      wavefunctions[:,i] = wavefunctions[:,i]/(np.linalg.norm(wavefunctions[:,i])*pm.deltax**0.5)

   # Make array of occupied wavefunctions
   occupied = np.zeros((pm.grid,pm.NE), dtype='complex')
   occupied_energies = np.zeros(pm.NE)
   for i in range(0,pm.NE):
      occupied[:,i] = wavefunctions[:,i]
      occupied_energies[i] = energies[i]

   # Make array of empty wavefunctions
   empty = np.zeros((pm.grid,pm.number), dtype='complex')
   empty_energies = np.zeros(pm.number)
   for i in range(0,pm.number):
      s = i + pm.NE
      empty[:,i] = wavefunctions[:,s]
      empty_energies[i] = energies[s]

   # Re-scale energies
   E1 = occupied_energies[-1] 
   E2 = empty_energies[0]
   E = (E2+E1)/2.0
   occupied_energies[:] = occupied_energies[:] - E
   empty_energies[:] = empty_energies[:] - E

   # Construct space-time grid
   st = SpaceTime()

   # Calculate the coulomb interaction in the time and domain
   print 'MBPT: computing coulomb interaction v'
   v = coulomb_interaction(st) 
   v_f = fourier(st,v,0)

   # Calculate the non-interacting green's function in imaginary time
   G0 = non_interacting_greens_function(st, occupied, occupied_energies, empty, empty_energies)
   Ga0 = alternate_greens_function(st, G0, empty, empty_energies)

   # Fourier transform the non-interacting green's function to the frequency domain
   G0_f = fourier(st,G0,0) 
   Ga0_f = fourier(st,Ga0,0)

   # Initial guess for the green's function
   print 'MBPT: constructing initial greens function G'
   G = np.zeros((st.tau_N_total,st.x_N,st.x_N), dtype='complex')
   Ga = np.zeros((st.tau_N_total,st.x_N,st.x_N), dtype='complex')
   G[:,:,:] = G0[:,:,:]
   Ga[:,:,:] = Ga0[:,:,:]

   # Determine level of self-consistency
   converged = False
   iteration = 0
   if(pm.self_consistent == 0):
      max_iterations = 1
   if(pm.self_consistent == 1):
      max_iterations = pm.max_iterations

   # GW self-consistency loop
   while(iteration < max_iterations and converged == False):
      if(iteration == 0 or pm.update_w == 1):
         if(pm.screening == 0):
            P = np.zeros((st.tau_N_total,st.x_N,st.x_N), dtype='complex') # Hartree-Fock Approximation
         else:
            P = irreducible_polarizability(st,G,Ga,iteration) # Calculate P in the time domain
         P_f = fourier(st,P,0) # Fourier transform to get P in the frequency domain
         W_f = screened_interaction(st,v_f,P_f,iteration) # Calculate W in the frequency domain
         W = fourier(st,W_f,1) # Fourier transform to get W in the time domain
      S = self_energy(st,G,W,iteration) # Calculate S in the time domain
      S = add_hartree(st,S,v,extract_density(st,G)) # Add the hartree potential to S in the time domain 
      S_f = fourier(st,S,0) # Fourier transform to get S in the frequency domain
      S_f = hedin_shift(st,S_f,occupied,empty) # Apply the hedin shift to S in the frequency domain
      if(pm.screening == 1 and pm.self_consistent == 1 and pm.update_w == 1): # Perform update of alternate G
         Sa = self_energy(st,Ga,W,iteration) # Calculate S in the time domain
         Sa = add_hartree(st,Sa,v,extract_density(st,G)) # Add the hartree potential to S in the time domain 
         Sa_f = fourier(st,Sa,0) # Fourier transform to get S in the frequency domain
         Sa_f = hedin_shift(st,Sa_f,occupied,empty) # Apply the hedin shift to S in the frequency domain
         Ga_f = dyson_equation(st,Ga0_f,Sa_f,iteration) # Solve the dyson equation in the frequency domain for the alternate green's function
         Ga = fourier(st,Ga_f,1) # Fourier transform to get Ga in the time domain
      G_f = dyson_equation(st,G0_f,S_f,iteration) # Solve the dyson equation in the frequency domain
      G = fourier(st,G_f,1) # Fourier transform to get G in the time domain
      if(iteration > 0):
         converged = has_converged(density,extract_density(st,G),iteration) # Test for converence
      density = extract_density(st,G) # Extract the ground-state density from G
      iteration += 1

   # Extract the ground-state density from G
   print
   print 'MBPT: computing density from the greens function G'
   density = extract_density(st,G)

   # Normalise the density
   density[:] = (density[:]*float(pm.NE))/(np.sum(density)*st.dx)

   # Output ground state density
   output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_mbpt_den.db','w')
   pickle.dump(density,output_file)
   output_file.close()

   # Output all hedin quantities
   #output_quantities(G0,P,W_f,S_f,G) # Uncomment this to save all hedin quantities to pickle files

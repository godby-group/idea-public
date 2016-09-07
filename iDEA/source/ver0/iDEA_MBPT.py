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
      self.tau_max = pm.tau_max
      self.tau_N = pm.tau_N
      self.dtau = float(2*pm.tau_max)/float(pm.tau_N-1)
      self.tau_grid = np.append(np.linspace(self.dtau/2.0,self.tau_max,self.tau_N/2),np.linspace(-self.tau_max,-self.dtau/2.0,self.tau_N/2))
      self.x_max = pm.xmax
      self.x_N = pm.grid
      self.dx = float(2*pm.xmax)/float(pm.grid-1)
      self.x_grid = np.linspace(-self.x_max,self.x_max,self.x_N)

# Function to construct the kinetic energy K
def constructK(st):
   K = -0.5*sps.diags([1, -2, 1],[-1, 0, 1], shape=(st.x_N,st.x_N), format='csr')/(st.dx**2)
   return K

# Function to construct the potential V
def constructV(st):
   if(pm.starting_orbitals == 'non'):
      Vdiagonal = []
      for i in xrange(0,len(st.x_grid)):
         Vdiagonal.append(pm.well(st.x_grid[i]))
      V = sps.spdiags(Vdiagonal, 0, st.x_N, st.x_N, format='csr')
   else:
      input_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(pm.starting_orbitals) + '_vks.db','r')
      Vdiagonal = pickle.load(input_file).real
      V = sps.spdiags(Vdiagonal, 0, st.x_N, st.x_N, format='csr')
   return V

# Function to construct the non-interacting green's function G0 in the time domain
def non_interacting_greens_function(st, occupied, occupied_energies, empty, empty_energies):
   G0 = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
   occupied_tensor = np.zeros((pm.NE,st.x_N,st.x_N), dtype='complex') 
   for i in xrange(0, st.x_N):
      for j in xrange(0, st.x_N):
         occupied_tensor[:,i,j] = occupied[i,:]*np.conjugate(occupied[j,:]) # phi_n(x_1)phi_n*(x_2) for occupied states
   empty_tensor = np.zeros((pm.number,st.x_N,st.x_N), dtype='complex') 
   for i in xrange(0, st.x_N):
      for j in xrange(0, st.x_N):
         empty_tensor[:,i,j] = empty[i,:]*np.conjugate(empty[j,:]) # phi_n(x_1)phi_n*(x_2) for empty states
   for k in xrange(0,st.tau_N):
      tau = st.tau_grid[k]
      string = 'MBPT: computing non-interacting greens function G0, tau = ' + str(tau)
      sprint.sprint(string,1,1,pm.msglvl)
      if(tau > 0.0): # Construct G0 for positive imaginary time
         for i in xrange(0,st.x_N):
            for j in xrange(0,st.x_N):
               G0[k,i,j] = -1.0j*np.sum(empty_tensor[:,i,j] * np.exp(-empty_energies[:]*tau))
      else: # Construct G0 for negative imaginary time
         for i in xrange(0,st.x_N):
            for j in xrange(0,st.x_N):
               G0[k,i,j] = 1.0j*np.sum(occupied_tensor[:,i,j] * np.exp(-occupied_energies[:]*tau))
   print
   return G0

# Function to calculate coulomb interaction in the frequency domain
def coulomb_interaction(st):
   v_f = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
   for i in xrange(0,st.x_N):
      for j in xrange(0,st.x_N):
         v_f[:,i,j] = 1.0/(abs(st.x_grid[j]-st.x_grid[i])+pm.acon) # Softened coulomb interaction
   return v_f

# Function to calculate the irreducible polarizability P in the time domain
def irreducible_polarizability(st,G,iteration):
   P = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
   for k in xrange(0,st.tau_N):
      for i in xrange(0,st.x_N):
         for j in xrange(0,st.x_N):
            P[k,i,j] = -1.0j*G[k,i,j]*G[-k,j,i]
   return P

# Function to calculate the screened interaction W_f in the frequency domain
def screened_interaction(st,v_f,P_f,iteration):
   W_f = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
   for k in xrange(0,st.tau_N): # At each frequency slice perform W = (I - v P)^-1 v
      W_f[k,:,:] = np.dot(npl.inv(np.eye(st.x_N,dtype='complex') - np.dot(v_f[k,:,:],P_f[k,:,:])*st.dx*st.dx),v_f[k,:,:])
   return W_f

# Function to calculate the self energy S in the time domain
def self_energy(st,G,W,iteration):
   S = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
   S[:,:,:] = 1.0j*G[:,:,:]*W[:,:,:] # GW approximation
   return S

# Function to correct diagrams of sigmal in the frequency domain
def correct_diagrams(st,S_f,v_f,density):
   V_h = np.dot(v_f[0,:,:],density)*st.dx
   V_hxc0 = np.zeros(st.x_N, dtype='complex')
   if(pm.starting_orbitals != 'non'):
      V_h0 = pickle.load(open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(pm.starting_orbitals) + '_vh.db','r'))
      V_xc0 = pickle.load(open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(pm.starting_orbitals) + '_vxc.db','r'))
      V_hxc0 = V_h0 + V_xc0 
   for i in xrange(0,st.x_N):
      S_f[:,i,i] += (V_h[i] - V_hxc0[i])/st.dx
   return S_f

# Function to perfrom the hedin shift in the frequency domain
def hedin_shift(st,S_f,occupied,empty):
   state = np.zeros(st.x_N, dtype='complex')
   state[:] = occupied[:,-1]
   expectation_value1 = np.vdot(state,np.dot(S_f[0,:,:],state.transpose()))*st.dx*st.dx # Calculates the expectation value of S(0) in the HOMO state
   state[:] = empty[:,0]
   expectation_value2 = np.vdot(state,np.dot(S_f[0,:,:],state.transpose()))*st.dx*st.dx # Calculates the expectation value of S(0) in the LUMO state
   expectation_value = (expectation_value1 + expectation_value2)/2.0 # Calculates the expectation value of S(0) at the fermi energy
   for i in xrange(0,st.x_N):
      S_f[:,i,i] = S_f[:,i,i] - expectation_value/st.dx
   return S_f

# Function to solve the dyson equation in the frequency domain to update G
def dyson_equation(st,G0_f,S_f,iteration):
   G_f = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex') # Greens function in the frequency domain
   for k in xrange(0,st.tau_N): # At each frequency slice perform G = (I - G0 S)^-1 G0
      G_f[k,:,:] = np.dot(npl.inv(np.eye(st.x_N,dtype='complex') - np.dot(G0_f[k,:,:],S_f[k,:,:])*st.dx*st.dx),G0_f[k,:,:])
   return G_f

# Function to generate the phase factors due to the offset time grid
def generate_phase_factors(st):
   phase_factors = np.zeros(st.tau_N, dtype='complex')
   for n in xrange(0,st.tau_N):
      phase_factors[n] = np.exp(-1.0j*math.pi*(n/st.tau_N))
   return phase_factors

# Function to fourier transform a given quantity
def fourier(st, A, inverse, phase_factors):
   transform =  np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
   if(inverse == 0):
      a = -1.0j*np.fft.fft(A,axis=0)*st.dtau
      for k in xrange(0,st.tau_N):
         transform[k,:,:] = a[k,:,:]*phase_factors[k]  
   if(inverse == 1):
      a = 1.0j*np.fft.ifft(A,axis=0)/st.dtau
      for k in xrange(0,st.tau_N):
         transform[k,:,:] = a[k,:,:]*np.conjugate(phase_factors[k])  
   return transform   

# Function to extract the ground-state density from G (using extrapolation to tau=0)
def extract_density(st,G):
   density = np.zeros(st.x_N, dtype='float')
   order = 10 # Order of polynomial used for fitting
   for j in xrange(0,st.x_N):
      x = []
      y = []
      for i in xrange(-order-1,0):
         x.append(st.tau_grid[i])
         y.append(G[i][j][j].imag)
      z = np.poly1d(np.polyfit(np.array(x), np.array(y), order))  
      density[j] = z(0)
   return density

# Function to test for convergence
def has_converged(density_new, density_old, iteration):
   convergence = abs(npl.norm(density_new-density_old))
   string = 'MBPT: performing self-consistency (iteration=' + str(iteration+1) + '): convergence = ' + str(convergence)
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

   # Construct space-time grid
   st = SpaceTime()

   # Construct the kinetic energy
   K = constructK(st)

   # Construct the potential
   V = constructV(st)

   # Constuct the hamiltonian
   H = K + V

   # Compute all wavefunctions
   print 'MBPT: computing eigenstates of single particle hamiltonian'
   solution = spla.eigsh(H, k=pm.grid-3, which='SA', maxiter=1000000)
   energies = solution[0] 
   wavefunctions = solution[1]

   # Normalise all wavefunctions
   length = len(wavefunctions[0,:])
   for i in xrange(0,length):
      wavefunctions[:,i] = wavefunctions[:,i]/(np.linalg.norm(wavefunctions[:,i])*pm.deltax**0.5)

   # Make array of occupied wavefunctions
   occupied = np.zeros((pm.grid,pm.NE), dtype='complex')
   occupied_energies = np.zeros(pm.NE)
   for i in xrange(0,pm.NE):
      occupied[:,i] = wavefunctions[:,i]
      occupied_energies[i] = energies[i]

   # Make array of empty wavefunctions
   empty = np.zeros((pm.grid,pm.number), dtype='complex')
   empty_energies = np.zeros(pm.number)
   for i in xrange(0,pm.number):
      s = i + pm.NE
      empty[:,i] = wavefunctions[:,s]
      empty_energies[i] = energies[s]

   # Re-scale energies
   E1 = occupied_energies[-1] 
   E2 = empty_energies[0]
   E = (E2+E1)/2.0
   occupied_energies[:] = occupied_energies[:] - E
   empty_energies[:] = empty_energies[:] - E

   # Compute phase factors for this grid
   phase_factors = generate_phase_factors(st)

   # Calculate the coulomb interaction in the frequency domain
   print 'MBPT: computing coulomb interaction v'
   v_f = coulomb_interaction(st) 

   # Calculate the non-interacting green's function in imaginary time
   G0 = non_interacting_greens_function(st, occupied, occupied_energies, empty, empty_energies)
   G0_f = fourier(st,G0,0,phase_factors) 

   # Initial guess for the green's function
   print 'MBPT: constructing initial greens function G'
   G = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
   G[:,:,:] = G0[:,:,:]

   # Determine level of self-consistency
   converged = False
   iteration = 0
   if(pm.self_consistent == 0):
      max_iterations = 1
   if(pm.self_consistent == 1):
      max_iterations = pm.max_iterations

   # GW self-consistency loop
   print 'MBPT: performing first iteration (one-shot)'
   while(iteration < max_iterations and converged == False):
      if(iteration == 0 or pm.update_w == 1):
         P = irreducible_polarizability(st,G,iteration) # Calculate P in the time domain
         P_f = fourier(st,P,0,phase_factors) # Fourier transform to get P in the frequency domain
         W_f = screened_interaction(st,v_f,P_f,iteration) # Calculate W in the frequency domain
         W = fourier(st,W_f,1,phase_factors) # Fourier transform to get W in the time domain
      S = self_energy(st,G,W,iteration) # Calculate S in the time domain
      S_f = fourier(st,S,0,phase_factors) # Fourier transform to get S in the frequency domain
      S_f = correct_diagrams(st,S_f,v_f,extract_density(st,G)) # Correct diagrams to S in the frequency domain
      S_f = hedin_shift(st,S_f,occupied,empty) # Apply the hedin shift to S in the frequency domain
      G_f = dyson_equation(st,G0_f,S_f,iteration) # Solve the dyson equation in the frequency domain
      G = fourier(st,G_f,1,phase_factors) # Fourier transform to get G in the time domain
      if(iteration > 0):
         converged = has_converged(density,extract_density(st,G),iteration) # Test for converence
      density = extract_density(st,G) # Extract the ground-state density from G
      iteration += 1

   # Extract the ground-state density from G
   if(pm.self_consistent == 1):
      print
   print 'MBPT: computing density from the greens function G'
   density = extract_density(st,G)

   # Normalise the density
   print 'MBPT: normalising density by ' + str(float(pm.NE)/(np.sum(density)*st.dx))
   density[:] = (density[:]*float(pm.NE))/(np.sum(density)*st.dx)

   # Output ground state density
   output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_mbpt_den.db','w')
   pickle.dump(density,output_file)
   output_file.close()

   # Output all hedin quantities
   #output_quantities(G0,P,W_f,S,G) # Uncomment this to save all hedin quantities to pickle files

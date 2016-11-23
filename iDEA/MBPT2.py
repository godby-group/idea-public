"""Computes Green function and self-energy in the GW approximation

Different flavours of GW (G0W0, scGW, QSGW) are available.  The implementation
follows the GW-space-time approach detailed in [Rojas1995]_ and  [Rieger1999]_.
"""
from __future__ import division
import copy
import numpy as np
import scipy as sp
import results as rs

class SpaceTimeGrid:
    """Stores spatial and frequency grids"""

    def __init__(self,pm):
        # (imaginary) time
        self.tau_max = pm.mbpt.tau_max
        self.tau_npt = pm.mbpt.tau_npt
        self.tau_delta = 2.0*self.tau_max / self.tau_npt

        # We use an offset grid for tau, since G0(it) is discontinuous at it=0.
        # The grid is laid out in a way appropriate for numpy's fft, and in
        # particular, this grid *always* starts with tau=dt/2.
        #   tau_grid = [dt/2,dt+dt/2,...,T/2+dt/2,-T/2+dt/2,...,-dt/2] # tau_npt odd
        #   tau_grid = [dt/2,dt+dt/2,...,T/2-dt/2,-T/2+dt/2,...,-dt/2] # tau_npt even
        self.tau_grid = 2*self.tau_max * np.fft.fftfreq(self.tau_npt) + self.tau_delta/2.0
        
        # (imaginary) frequency
        self.omega_max = np.pi / self.tau_delta
        # no shift for the frequency grid
        self.omega_grid = 2*self.omega_max * np.fft.fftfreq(self.tau_npt)
        self.omega_delta = 2*np.pi / 2*self.tau_max

        # space
        self.x_max = pm.sys.xmax
        self.x_npt = pm.sys.grid
        self.x_delta = float(2*pm.sys.xmax)/float(pm.sys.grid-1)
        self.x_grid = np.linspace(-self.x_max,self.x_max,self.x_npt)

        # orbitals
        self.norb = pm.mbpt.norb
        self.NE = pm.sys.NE

    def __str__(self):
        """Returns string with description of main parameters"""
        s = ""
        s += "Spatial grid: [{:.3f}, {:.3f}] in steps of dx = {:.3f}\n".\
                format(-self.x_max,self.x_max,self.x_delta)
        s += "Temporal grid: [{:.3f}, {:.3f}] in steps of dt = {:.3f}\n".\
                format(-self.tau_max,self.tau_max,self.tau_delta)
        s += "Orbitals: {} occupied, {} empty ({:.1f}% of basis set)\n".\
                format(self.NE, self.norb - self.NE, self.norb/self.x_npt*100)
        return s

        
def main(parameters):
    """Runs GW calculation"""
    pm = parameters
    results = rs.Results()

    st = SpaceTimeGrid(pm)
    pm.sprint(str(st))

    # read in eigenvalues and eigenfunctions of
    # starting Hamiltonian
    h0_energies = rs.Results.read('gs_{}_eigv'.format(pm.mbpt.h0), pm)
    h0_orbitals = rs.Results.read('gs_{}_eigf'.format(pm.mbpt.h0), pm)

    norb = len(h0_energies)
    if norb < st.norb:
        raise ValueError("Not enough orbitals: {} computed, {} requested.".format(norb,st.norb))
    else:
        h0_energies = h0_energies[:st.norb]
        h0_orbitals = h0_orbitals[:st.norb]

    homo = h0_energies[st.NE-1]
    lumo = h0_energies[st.NE]
    gap = lumo - homo
    e_fermi = homo + gap / 2
    pm.sprint('MBPT: Shifting center of gap from {:.3f} Ha to zero.'.format(e_fermi))
    h0_energies -= e_fermi
    results.add(h0_energies, name="gs_mbpt_eigv0")
    results.add(h0_orbitals, name="gs_mbpt_eigf0")
    results.add(e_fermi, name="gs_mbpt_efermi0")

    # check that G(it) is well described
    exp_factor = np.exp(-(lumo-e_fermi)*st.tau_max)
    if exp_factor > 1e-2:
        t2 = -np.log(1e-2)/(lumo-e_fermi)
        t3 = -np.log(1e-3)/(lumo-e_fermi)
        pm.sprint("Warning: Width of tau-grid for G(it) is too small for HOMO-LUMO "\
              "gap {:.3f} Ha.\n Increase tau_max to {:.1f} for decay to 1e-2 "\
              "or {:.1f} for decay to 1e-3".format(gap,t2,t3))


    pm.sprint('MBPT: setting up G0(it)')
    G0 = non_interacting_green_function(h0_orbitals, h0_energies, st)

    if pm.mbpt.save_diag:
        exp_values = bracket_r(G0, h0_orbitals,st)
        results.add(exp_values,"gs_mbpt_G0_dg")

    #if pm.mbpt.flavour == 'G0W0':
    #    results = G0W0(pm, results)
    #elif pm.mbpt.flavour == 'GW':
    #    results = SCGW(pm, results)
    #elif pm.mbpt.flavour == 'QSGW':
    #    results = QSGW_imaginary_time(orbitals, sp_energies, pm, results)
    #else:
    #    raise ValueError("MBPT mode {} is not implemented".format(pm.mbpt.flavour))

    return results


def non_interacting_green_function(orbitals, energies, st, zero='0-'):
    r"""Calculates non-interacting Green function G0(r,r';it).

    :math:`G_0(r,r';i\tau)` is constructed from a set of eigenvectors
    and eigenenergies of a single-particle Hamiltonian.

    .. math ::
        G_0(r,r';i\tau) = (-i) \sum_s^{empty} \varphi_s(r) \varphi_s(r') e^{-\varepsilon_s\tau} \theta(\tau)
                        +   i  \sum_s^{occupied} \varphi_s(r) \varphi_s(r') e^{-\varepsilon_s\tau} \theta(-\tau)


    See equation 3.3 of [Rieger1999]_. Note that we have reversed the sign of
    :math:`\tau` in order to be consistent with Hedin [Hedin1970]_.

    FLOPS: norb * (grid**2 + 2 * tau_npt * grid**2)

    parameters
    ----------
    orbitals : array
        set of single-particle orbitals
    energies : array
        corresponding single-particle energies
    st : object
        contains space-time parameters
    """
    coef = np.zeros((st.norb,st.tau_npt), dtype=complex)
    for i in range(st.norb):
        en = energies[i]

        # note: this could still be vectorized
        for k in range(0,st.tau_npt):
            tau = st.tau_grid[k]

            if en > 0 and tau > 0:
                # tau > 0, empty states
                coef[i,k] = -1J * np.exp(-en * tau)
            elif en < 0 and tau < 0:
                # tau < 0, occupied states
                coef[i,k] = +1J * np.exp(-en * tau)

    # one call to np.dot with reshape
    orb_mat = np.empty( (st.x_npt, st.x_npt, st.norb), dtype=complex )
    for i in range(st.norb):
        orb = orbitals[i]
        orb_mat[:,:,i] = np.tensordot(orb.conj(), orb,axes=0)
    # for some reason, einsum is significantly slower...
    orb_mat_r = orb_mat.reshape(st.x_npt*st.x_npt, st.norb)
    #G0 = np.einsum('ij,jl->il',orb_mat.reshape(st.x_npt*st.x_npt, norb),coef)
    G0 = np.dot(orb_mat_r,coef).reshape(st.x_npt,st.x_npt,st.tau_npt)

    return G0


def bracket_r(O, orbitals, st, mode='diagonal'):
    r"""Calculate expectationvalues of O(r,r';t) for each t wrt orbitals

    .. math:: O_{jj}(t) = \langle \varphi_j | O(r,r';t) | \varphi_j\rangle
                        = \int \varphi_j^*(r) O(r,r';t)\varphi_j(r')\,dr\,dr'

    Note: For non-hermitian O, the pairing of r,r' with \varphi_j^*,\varphi_j
    matters.

    parameters
    ----------
    O: array
        Operator O(r,r';t)
    orbitals: array
        Array of shape (norb, grid) containing orbtials
    pm: object
        Parameters object
    mode: string
        if 'diagonal', evaluates <j|O|j> for all j
        if 'matrix', evaluates <i|O|j> for all i,j

    Returns
        bracket_r[i,j]:  i=orbital index, j=index of temporal grid
    """
    orbs = copy.copy(orbitals) * st.x_delta  # factor needed for integration

    # Performing one matrix-matrix multiplication is substantially faster than
    # performing t matrix-vector multiplications.
    # Note: If we do not np.reshape, np.dot internally performs matrix-vector
    # multiplications here.

    # numpy.dot(a,b) sums over last axis of a and 2nd-to-last axis of b
    # tmp.shape == (norb, x_npt*tau_npt)
    tmp = np.dot(orbs, O.reshape((st.x_npt, st.x_npt * st.tau_npt)))

    if mode == 'diagonal':
        # Then, we do element-wise multiplication + summation over the grid axis
        # to get the scalar product.
        bracket_r = (orbs.conj()[:,:,None] * tmp.reshape((st.norb,st.x_npt,st.tau_npt))).sum(1)
    elif mode == 'matrix':
        # bracket_r.shape == (norb,norb,t)
        bracket_r  = np.dot(orbs.conj(), tmp.reshape((st.norb,st.x_npt,st.tau_npt)))
    else:
        raise ValueError("Unknown mode {}".format(mode))
        
    return bracket_r


#
## Function to construct the kinetic energy K
#def constructK(st):
#   K = -0.5*sps.diags([1, -2, 1],[-1, 0, 1], shape=(st.x_N,st.x_N), format='csr')/(st.dx**2)
#   return K
#
## Function to construct the potential V
#def constructV(st):
#   if(pm.mbpt.starting_orbitals == 'non'):
#      Vdiagonal = []
#      for i in xrange(0,len(st.x_grid)):
#         Vdiagonal.append(pm.sys.v_ext(st.x_grid[i]))
#      V = sps.spdiags(Vdiagonal, 0, st.x_N, st.x_N, format='csr')
#   else:
#      name = 'gs_{}_vks'.format(pm.mbpt.starting_orbitals)
#      data = rs.Results.read(name, pm)
#      #input_file = open('outputs/' + str(pm.run.name) + '/raw/' + str(pm.run.name) + '_' + str(pm.sys.NE) + 'gs_' + str(pm.mbpt.starting_orbitals) + '_vks.db','r')
#      Vdiagonal = data.real
#      V = sps.spdiags(Vdiagonal, 0, st.x_N, st.x_N, format='csr')
#   return V
#
## Function to construct the non-interacting green's function G0 in the time domain
#def non_interacting_greens_function(st, occupied, occupied_energies, empty, empty_energies):
#   G0 = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   occupied_tensor = np.zeros((pm.sys.NE,st.x_N,st.x_N), dtype='complex') 
#   for i in xrange(0, st.x_N):
#      for j in xrange(0, st.x_N):
#         occupied_tensor[:,i,j] = occupied[i,:]*np.conjugate(occupied[j,:]) # phi_n(x_1)phi_n*(x_2) for occupied states
#   empty_tensor = np.zeros((pm.mbpt.number_empty,st.x_N,st.x_N), dtype='complex') 
#   for i in xrange(0, st.x_N):
#      for j in xrange(0, st.x_N):
#         empty_tensor[:,i,j] = empty[i,:]*np.conjugate(empty[j,:]) # phi_n(x_1)phi_n*(x_2) for empty states
#   for k in xrange(0,st.tau_N):
#      tau = st.tau_grid[k]
#      string = 'MBPT: computing non-interacting greens function G0, tau = ' + str(tau)
#      pm.sprint(string,1,newline=False)
#      if(tau > 0.0): # Construct G0 for positive imaginary time
#         for i in xrange(0,st.x_N):
#            for j in xrange(0,st.x_N):
#               G0[k,i,j] = -1.0j*np.sum(empty_tensor[:,i,j] * np.exp(-empty_energies[:]*tau))
#      else: # Construct G0 for negative imaginary time
#         for i in xrange(0,st.x_N):
#            for j in xrange(0,st.x_N):
#               G0[k,i,j] = 1.0j*np.sum(occupied_tensor[:,i,j] * np.exp(-occupied_energies[:]*tau))
#   print
#   return G0
#
## Function to calculate coulomb interaction in the frequency domain
#def coulomb_interaction(st):
#   v_f = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   for i in xrange(0,st.x_N):
#      for j in xrange(0,st.x_N):
#         v_f[:,i,j] = 1.0/(abs(st.x_grid[j]-st.x_grid[i])+pm.sys.acon) # Softened coulomb interaction
#   return v_f
#
## Function to calculate the irreducible polarizability P in the time domain
#def irreducible_polarizability(st,G,iteration):
#   P = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   for k in xrange(0,st.tau_N):
#      for i in xrange(0,st.x_N):
#         for j in xrange(0,st.x_N):
#            P[k,i,j] = -1.0j*G[k,i,j]*G[-k-1,j,i]
#   return P
#
## Function to calculate the screened interaction W_f in the frequency domain
#def screened_interaction(st,v_f,P_f,iteration):
#   W_f = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   for k in xrange(0,st.tau_N): # At each frequency slice perform W = (I - v P)^-1 v
#      W_f[k,:,:] = np.dot(npl.inv(np.eye(st.x_N,dtype='complex') - np.dot(v_f[k,:,:],P_f[k,:,:])*st.dx*st.dx),v_f[k,:,:])
#   return W_f
#
## Function to calculate the self energy S in the time domain
#def self_energy(st,G,W,iteration):
#   S = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   S[:,:,:] = 1.0j*G[:,:,:]*W[:,:,:] # GW approximation
#   return S
#
## Function to correct diagrams of sigmal in the frequency domain
#def correct_diagrams(st,S_f,v_f,density):
#   V_h = np.dot(v_f[0,:,:],density)*st.dx
#   V_hxc0 = np.zeros(st.x_N, dtype='complex')
#   if(pm.mbpt.starting_orbitals != 'non'):
#      name = 'gs_{}_vh'.format(pm.mbpt.starting_orbitals)
#      V_h0 = rs.Results.read(name, pm)
#      name = 'gs_{}_vxc'.format(pm.mbpt.starting_orbitals)
#      V_xc0 = rs.Results.read(name, pm)
#      V_hxc0 = V_h0 + V_xc0 
#   for i in xrange(0,st.x_N):
#      S_f[:,i,i] += (V_h[i] - V_hxc0[i])/st.dx
#   return S_f
#
## Function to perfrom the hedin shift in the frequency domain
#def hedin_shift(st,S_f,occupied,empty):
#   state = np.zeros(st.x_N, dtype='complex')
#   state[:] = occupied[:,-1]
#   expectation_value1 = np.vdot(state,np.dot(S_f[0,:,:],state.transpose()))*st.dx*st.dx # Calculates the expectation value of S(0) in the HOMO state
#   state[:] = empty[:,0]
#   expectation_value2 = np.vdot(state,np.dot(S_f[0,:,:],state.transpose()))*st.dx*st.dx # Calculates the expectation value of S(0) in the LUMO state
#   expectation_value = (expectation_value1 + expectation_value2)/2.0 # Calculates the expectation value of S(0) at the fermi energy
#   for i in xrange(0,st.x_N):
#      S_f[:,i,i] = S_f[:,i,i] - expectation_value/st.dx
#   return S_f
#
## Function to solve the dyson equation in the frequency domain to update G
#def dyson_equation(st,G0_f,S_f,iteration):
#   G_f = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex') # Greens function in the frequency domain
#   for k in xrange(0,st.tau_N): # At each frequency slice perform G = (I - G0 S)^-1 G0
#      G_f[k,:,:] = np.dot(npl.inv(np.eye(st.x_N,dtype='complex') - np.dot(G0_f[k,:,:],S_f[k,:,:])*st.dx*st.dx),G0_f[k,:,:])
#   return G_f
#
## Function to generate the phase factors due to the offset time grid
#def generate_phase_factors(st):
#   phase_factors = np.zeros(st.tau_N, dtype='complex')
#   for n in xrange(0,st.tau_N):
#      phase_factors[n] = np.exp(-1.0j*np.pi*(n/st.tau_N))
#   return phase_factors
#
## Function to fourier transform a given quantity
#def fourier(st, A, inverse, phase_factors):
#   transform =  np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   if(inverse == 0):
#      a = -1.0j*np.fft.fft(A,axis=0)*st.dtau
#      for k in xrange(0,st.tau_N):
#         transform[k,:,:] = a[k,:,:]*phase_factors[k]  
#   if(inverse == 1):
#      a = 1.0j*np.fft.ifft(A,axis=0)/st.dtau
#      for k in xrange(0,st.tau_N):
#         transform[k,:,:] = a[k,:,:]*np.conjugate(phase_factors[k])  
#   return transform   
#
## Function to extract the ground-state density from G (using extrapolation to tau=0)
#def extract_density(st,G):
#   density = np.zeros(st.x_N, dtype='float')
#   order = 10 # Order of polynomial used for fitting
#   for j in xrange(0,st.x_N):
#      x = []
#      y = []
#      for i in xrange(-order-1,0):
#         x.append(st.tau_grid[i])
#         y.append(G[i][j][j].imag)
#      z = np.poly1d(np.polyfit(np.array(x), np.array(y), order))  
#      density[j] = z(0)
#   return density
#
## Function to test for convergence
#def has_converged(density_new, density_old, iteration):
#   convergence = abs(npl.norm(density_new-density_old))
#   string = 'MBPT: performing self-consistency (iteration=' + str(iteration+1) + '): convergence = ' + str(convergence)
#   pm.sprint(string,1,newline=False)
#   if(convergence < pm.mbpt.tolerance):
#      return True
#   else:
#      return False
#
## Function to save all hedin quantities to pickle files
#def output_quantities(G0,G0_f,P,P_f,Sc_f,S_f,G):
#   output_file = open('outputs/' + str(pm.run.name) + '/data/g0.db','w')
#   pickle.dump(G0,output_file)
#   output_file.close()
#   output_file = open('outputs/' + str(pm.run.name) + '/data/g0_f.db','w')
#   pickle.dump(G0_f,output_file)
#   output_file.close()
#   output_file = open('outputs/' + str(pm.run.name) + '/data/p.db','w')
#   pickle.dump(P,output_file)
#   output_file.close()
#   output_file = open('outputs/' + str(pm.run.name) + '/data/p_f.db','w')
#   pickle.dump(P_f,output_file)
#   output_file.close()
#   output_file = open('outputs/' + str(pm.run.name) + '/data/sc_f.db','w')
#   pickle.dump(Sc_f,output_file)
#   output_file.close()
#   output_file = open('outputs/' + str(pm.run.name) + '/data/s_f.db','w')
#   pickle.dump(S_f,output_file)
#   output_file.close()
#   output_file = open('outputs/' + str(pm.run.name) + '/data/g.db','w')
#   pickle.dump(G,output_file)
#   output_file.close()
#
## Main function
#def main(parameters):
#   global pm
#   pm = parameters
#
#   # Construct space-time grid
#   st = SpaceTime()
#   
#   # Construct the kinetic energy
#   K = constructK(st)
#
#   # Construct the potential
#   V = constructV(st)
#
#   # Constuct the hamiltonian
#   H = K + V 
#
#   # Compute all wavefunctions
#   print 'MBPT: computing eigenstates of single particle hamiltonian'
#   solution = spsla.eigsh(H, k=pm.sys.grid-3, which='SA', maxiter=1000000)
#   energies = solution[0] 
#   wavefunctions = solution[1]
#
#   # Normalise all wavefunctions
#   length = len(wavefunctions[0,:])
#   for i in xrange(0,length):
#      wavefunctions[:,i] = wavefunctions[:,i]/(np.linalg.norm(wavefunctions[:,i])*pm.sys.deltax**0.5)
#
#   # Make array of occupied wavefunctions
#   occupied = np.zeros((pm.sys.grid,pm.sys.NE), dtype='complex')
#   occupied_energies = np.zeros(pm.sys.NE)
#   for i in xrange(0,pm.sys.NE):
#      occupied[:,i] = wavefunctions[:,i]
#      occupied_energies[i] = energies[i]
#
#   # Make array of empty wavefunctions
#   empty = np.zeros((pm.sys.grid,pm.mbpt.number_empty), dtype='complex')
#   empty_energies = np.zeros(pm.mbpt.number_empty)
#   for i in xrange(0,pm.mbpt.number_empty):
#      s = i + pm.sys.NE
#      empty[:,i] = wavefunctions[:,s]
#      empty_energies[i] = energies[s]
#
#   # Re-scale energies
#   E1 = occupied_energies[-1] 
#   E2 = empty_energies[0]
#   E = (E2+E1)/2.0
#   occupied_energies[:] = occupied_energies[:] - E
#   empty_energies[:] = empty_energies[:] - E
#
#   # Compute phase factors for this grid
#   phase_factors = generate_phase_factors(st)
#
#   # Calculate the coulomb interaction in the frequency domain
#   print 'MBPT: computing coulomb interaction v'
#   v_f = coulomb_interaction(st) 
#
#   # Calculate the non-interacting green's function in imaginary time
#   G0 = non_interacting_greens_function(st, occupied, occupied_energies, empty, empty_energies)
#   G0_f = fourier(st,G0,0,phase_factors) 
#
#   # Initial guess for the green's function
#   print 'MBPT: constructing initial greens function G'
#   G = np.zeros((st.tau_N,st.x_N,st.x_N), dtype='complex')
#   G[:,:,:] = G0[:,:,:]
#
#   # Determine level of self-consistency
#   converged = False
#   iteration = 0
#   if(pm.mbpt.self_consistent == 0):
#      max_iterations = 1
#   if(pm.mbpt.self_consistent == 1):
#      max_iterations = pm.mbpt.max_iterations
#
#   # GW self-consistency loop
#   print 'MBPT: performing first iteration (one-shot)'
#   while(iteration < max_iterations and converged == False):
#      if(iteration == 0 or pm.mbpt.update_w == True):
#         P = irreducible_polarizability(st,G,iteration) # Calculate P in the time domain
#         P_f = fourier(st,P,0,phase_factors) # Fourier transform to get P in the frequency domain
#         W_f = screened_interaction(st,v_f,P_f,iteration) # Calculate W in the frequency domain
#         W = fourier(st,W_f,1,phase_factors) # Fourier transform to get W in the time domain
#      S = self_energy(st,G,W,iteration) # Calculate S in the time domain
#      S_f = fourier(st,S,0,phase_factors) # Fourier transform to get S in the frequency domain
#      S_f = correct_diagrams(st,S_f,v_f,extract_density(st,G)) # Correct diagrams to S in the frequency domain
#      S_f = hedin_shift(st,S_f,occupied,empty) # Apply the hedin shift to S in the frequency domain
#      G_f = dyson_equation(st,G0_f,S_f,iteration) # Solve the dyson equation in the frequency domain
#      G = fourier(st,G_f,1,phase_factors) # Fourier transform to get G in the time domain
#      if(iteration > 0):
#         converged = has_converged(density,extract_density(st,G),iteration) # Test for converence
#      density = extract_density(st,G) # Extract the ground-state density from G
#      iteration += 1
#
#   # Extract the ground-state density from G
#   if(pm.mbpt.self_consistent == 1):
#      print
#   print 'MBPT: computing density from the greens function G'
#   density = extract_density(st,G)
#
#   # Normalise the density
#   print 'MBPT: normalising density by ' + str(float(pm.sys.NE)/(np.sum(density)*st.dx))
#   density[:] = (density[:]*float(pm.sys.NE))/(np.sum(density)*st.dx)
#
#   # Output ground state density
#   results = rs.Results()
#   results.add(density,'gs_mbpt_den')
#   if pm.run.save:
#      results.save(pm)
#      
#   # Output all hedin quantities
#   if(pm.mbpt.output_hedin == True):
#      output_quantities(G0,G0_f,P,P_f,W_f-v_f,S_f,G) 
#   
#   return results

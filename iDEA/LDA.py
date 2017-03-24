"""Computes approximations to VKS, VH, VXC using the LDA self consistently. For ground state calculations the code outputs the LDA orbitals 
and energies of the system, the ground-state charge density and the Kohn-Sham potential. For time dependent calculation the code also outputs 
the time-dependent charge and current densities anjd the time-dependent Kohn-Sham potential. Note: Uses the [adiabatic] local density 
approximations ([A]LDA) to calculate the [time-dependent] electron density [and current] for a system of N electrons.
"""


import pickle
import numpy as np
import scipy as sp
import copy as copy
import RE_Utilities
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
import results as rs
import mix
import minimize


def groundstate(pm, H):
   r"""Calculates the oribitals and ground state density for the system for a given potential

    .. math:: H \psi_{i} = E_{i} \psi_{i}
                       
   parameters
   ----------
   v_KS : array_like
        KS potential

   returns array_like, array_like, array_like
        density, normalised orbitals indexed as eigf[space_index,orbital_number], energies
   """	
   
   #e,eigf = spsla.eigsh(H, k=pm.sys.grid/2, which='SA') 
   e,eigf = spla.eigh(H)
   #e,eigf = spla.eig_banded(H,True) 
   
   eigf /= np.sqrt(pm.sys.deltax)
   n = electron_density(pm, eigf)

   return n,eigf,e

def electron_density(pm, orbitals):
    r"""Compute density for given orbitals

    parameters
    ----------
    orbitals: array_like
      array of properly normalised orbitals[space_index,orital_number]

    returns
    -------
    n: array_like
      electron density
    """
    occupied = orbitals[:, :pm.sys.NE]
    n = np.sum(occupied*occupied.conj(), axis=1)
    return n


def construct_hamiltonian(pm, v_KS):
    r"""Compute LDA Hamiltonian

    Computes LDA Hamiltonian from a given Kohn-Sham potential.

    parameters
    ----------
    v_KS : array_like
         KS potential

    returns array_like
         Hamiltonian matrix
    """
    T = -0.5 * sps.diags([1,-2,1],[-1,0,1], shape=(pm.sys.grid, pm.sys.grid), dtype=np.float, format='csr') / pm.sys.deltax**2
    V = sps.diags(v_KS, 0, shape=(pm.sys.grid, pm.sys.grid), dtype=np.float, format='csr')

    # banded version
    #H = np.zeros((2,pm.sys.grid),dtype='float')
    ## 3-point stencil for 2nd derivative
    #H[0,:] = np.ones(pm.sys.grid)/pm.sys.deltax**2
    #H[1,:] = -0.5*np.ones(pm.sys.grid)/pm.sys.deltax**2 
    #H[0,:] += v_KS[:]

    return (T+V).toarray()
    
def update_hamiltonian(pm, H, v_KS):
    r"""Update LDA Hamiltonian with new KS potential

    parameters
    ----------
    H : array_like
         old Hamiltonian matrix
    v_KS : array_like
         KS potential

    returns array_like
         Hamiltonian matrix
    """
    for i in range(pm.sys.grid):
        H[i,i] = -0.5 * (-2.0) / pm.sys.deltax**2 + v_KS[i]

    return H


def hartree_potential(pm, density):
   r"""Computes Hartree potential for a given density

   .. math::

       V_H(r) = = \int U(r,r') n(r')dr'

   parameters
   ----------
   density : array_like
        given density

   returns array_like
   """
   return np.dot(pm.space.v_int,density)*pm.sys.deltax

def hartree_energy(pm, V_H, density):
   r"""Computes Hartree energy for a given density

   .. math::

       E_H[n] = \frac{1}{2} \int n(r) V_H(r) dr

   parameters
   ----------
   V_H : array_like
        Hartree potential
        
   density : array_like
        given density

   returns array_like
   """
   return 0.5 * np.dot(V_H,density)*pm.sys.deltax

# these are Mike's parameters for finite LDAs
elda = {}  # parameters for \varepsilon_{xc}(n)
elda[1] = {
   'a' :  -0.803,
   'b' :  0.82,
   'c' :  -0.47,
   'd' :  0.638,
}
elda[2] =  {
   'a' :  -0.74,
   'b' :  0.68,
   'c' :  -0.38,
   'd' :  0.604,
}
elda['n'] =  {
   'a' : -0.77,
   'b' : 0.79,
   'c' : -0.48,
   'd' : 0.61,
}

vlda = {}  # parameters for V_{xc}(n)
for n in [1,2,'n']:
    eps = elda[n]
    a = eps['a']
    b = eps['b']
    c = eps['c']
    d = eps['d']

    vlda[n] = { 
      'a': (d+1)*a,
      'b': (d+2)*b,
      'c': (d+3)*c,
      'd': d
    }

dlda = {} # parameters for dV_{xc}(n)/dn
for n in [1,2,'n']:
    v = vlda[n]
    a = v['a']
    b = v['b']
    c = v['c']
    d = v['d']

    dlda[n] = { 
      'a': d*a,
      'b': (d+1)*b,
      'c': (d+2)*c,
      'd': d-1
    }


def VXC(pm , Den): 
   r"""Finite LDA approximation for the Exchange-Correlation potential

   parameters
   ----------
   Den : array_like
        density

   returns array_like
        Exchange-Correlation potential
   """
   V_xc = np.zeros(pm.sys.grid,dtype='float')

   NE = pm.sys.NE
   if NE > 2:
       NE = 'n'
   p = vlda[NE]

   V_xc = (p['a'] + p['b'] * Den + p['c'] * Den**2) * Den**p['d']

   return V_xc

def DXC(pm , n): 
   r"""Derivative of finite LDA for the Exchange-Correlation potential

   This function simply returns the derivative of VXC

   parameters
   ----------
   n : array_like
        density

   returns array_like
        Exchange-Correlation potential
   """
   D_xc = np.zeros(pm.sys.grid,dtype='float')

   NE = pm.sys.NE
   if NE > 2:
       NE = 'n'
   p = dlda[NE]

   D_xc = (p['a'] + p['b'] * n + p['c'] * n**2) * n**p['d']
   #D_xc = (p['b'] + 2 * p['c'] * n) * n**p['d'] \
   #     + (p['a'] + p['b'] * n + p['c'] * n**2) * p['d'] * n**(p['d']-1)

   return D_xc 


def EXC(pm, n): 
   r"""Finite LDA approximation for the Exchange-Correlation energy

   .. math ::
        E_{xc} = \int \varepsilon_{xc}(n(r)) n(r) dr

   parameters
   ----------
   n : array_like
        density

   returns float
        Exchange-Correlation energy
   """

   NE = pm.sys.NE
   if NE > 2:
       NE = 'n'
   p = elda[NE]

   e_xc = (p['a'] + p['b'] * n + p['c'] * n**2) * n**p['d']
   E_xc_LDA = np.dot(e_xc, n) * pm.sys.deltax

   return E_xc_LDA


def total_energy_eigv(pm, eigv, eigf=None, n=None, V_H=None, V_xc=None):
   r"""Calculates the total energy of the self-consistent LDA density                 
   Uses knowledge of Kohn-Sham eigenvalues

   .. math ::
       E = \sum_i \varepsilon_i + E_{xc}[n] - E_H[n] - \int \rho(r) V_{xc}(r)dr


   parameters
   ----------
   pm : array_like
        external potential
   eigv : array_like
        eigenvalues
   eigf : array_like
        eigenfunctions
   n : array_like
        density
   V_H : array_like
        Hartree potential
   V_xc : array_like
        exchange-correlation potential

   returns float
   """		

   if n is None:
       if eigf is None:
           raise ValueError("Need to specify either n or eigf")
       else:
           n = electron_density(pm, eigf)
   if not V_H:
       V_H = hartree_potential(pm, n)
   if not V_xc:
       V_xc = VXC(pm,n)
   
   E_LDA = 0.0
   for i in range(pm.sys.NE):
      E_LDA += eigv[i]

   E_LDA -= hartree_energy(pm, V_H, n)
   E_LDA -= np.dot(n, V_xc) * pm.sys.deltax
   E_LDA += EXC(pm, n)

   return E_LDA.real

def total_energy_eigf(pm, eigf, n=None, V_H=None):
   r"""Calculates the total energy of the self-consistent LDA density                 
   Uses Kohn-Sham wave functions only

   .. math::
       E = \sum_i \langle \psi_i | T | \psi_i\rangle + E_H[n] + E_{xc}[n] + \int \rho(r) V_{ext}(r)dr

   parameters
   ----------
   pm : array_like
        external potential
   eigf : array_like
        eigenfunctions
   density : array_like
        density
   V_H : array_like
        Hartree potential

   returns float
   """		

   if n is None:
       n = electron_density(pm, eigf)
   if V_H is None:
       V_H = hartree_potential(pm, n)

   E_LDA = 0.0
   E_LDA += kinetic_energy(pm, eigf)
   E_LDA += hartree_energy(pm, V_H, n)
   E_LDA += EXC(pm, n)
   E_LDA += np.dot(pm.space.v_ext, n) * pm.space.delta

   return E_LDA.real
  

def kinetic_energy(pm, eigf):
    r"""Compute kinetic energy of orbitals

    Note: With our (perhaps slightly strange definition) of H, we have
    <psi|H|psi> = psi^T H psi *dx

    TODO: This would be much cheaper in reciprocal space

    parameters
    ----------
    eigf: array_like
      (grid, nwf) eigen functions
    """
    T = -0.5 * sps.diags([1,-2,1],[-1,0,1], shape=(pm.sys.grid, pm.sys.grid), dtype=np.float, format='csr') / pm.sys.deltax**2

    occ = eigf[:,:pm.sys.NE]
    energies = (occ.conj() * T.dot(occ)).sum(0) * pm.sys.deltax

    return np.sum(energies)


   
def CalculateCurrentDensity(pm, n, j):
   r"""Calculates the current density of a time evolving wavefunction by solving the continuity equation.

   .. math::

       \frac{\partial n}{\partial t} + \nabla \cdot j = 0
       
   Note: This function requires RE_Utilities.so

   parameters
   ----------
   total_td_density : array_like
      Time dependent density of the system indexed as total_td_density[time_index][space_index]

   returns array_like
      Time dependent current density indexed as current_density[time_index][space_index]
   """
   J = RE_Utilities.continuity_eqn(pm.sys.grid,pm.sys.deltax,pm.sys.deltat,n[j,:],n[j-1,:])
   if pm.sys.im == 1:
      for j in xrange(pm.sys.grid):
         for k in xrange(j+1):
            x = k*pm.sys.deltax-pm.sys.xmax
            J[j] -= abs(pm.sys.v_pert_im(x))*n[j,k]*pm.sys.deltax
   return J


def CrankNicolson(pm, v, Psi, n, j): 
   r"""Solves Crank Nicolson Equation
   """
   Mat = LHS(pm, v,j)
   Mat = Mat.tocsr()
   Matin =- (Mat-sps.identity(pm.sys.grid,dtype='complex'))+sps.identity(pm.sys.grid,dtype='complex')
   for i in range(pm.sys.NE):
      B = Matin*Psi[i,j-1,:]
      Psi[i,j,:] = spsla.spsolve(Mat,B)
      n[j,:] = 0
      for i in range(pm.sys.NE):
         n[j,:] += abs(Psi[i,j,:])**2
   return n,Psi


def LHS(pm, v, j):	
   r"""Constructs the matrix A to be used in the crank-nicholson solution of Ax=b when evolving the wavefunction in time (Ax=b)
   """
   CNLHS = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex') # Matrix for the left hand side of the Crank Nicholson method										
   for i in xrange(pm.sys.grid):
      CNLHS[i,i] = 1.0+0.5j*pm.sys.deltat*(1.0/pm.sys.deltax**2+v[j,i])
      if i < pm.sys.grid-1:
         CNLHS[i,i+1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax**2)
      if i > 0:
         CNLHS[i,i-1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax**2)
   return CNLHS

        

# Main function
def main(parameters):
   r"""Performs LDA calculation

   parameters
   ----------
   parameters : object
      Parameters object

   returns object
      Results object
   """
   pm = parameters
   
   v_ext = np.zeros(pm.sys.grid,dtype='float')
   Psi = np.zeros((pm.sys.NE,pm.sys.imax,pm.sys.grid), dtype=np.complex)
   for i in xrange(pm.sys.grid):
      v_ext[i] = pm.sys.v_ext((i*pm.sys.deltax-pm.sys.xmax)) # External potential

   # take external potential for initial guess
   H = construct_hamiltonian(pm, v_ext)
   n,waves,energies = groundstate(pm, H)
   U = pm.space.v_int # Coulomb matrix
   convergence = 2*pm.sys.NE  # maximum density difference
   iteration = 1

   if pm.lda.mix_type == 'pulay':
       mixer = mix.PulayMixer(pm, order=pm.lda.pulay_order, preconditioner=pm.lda.preconditioner)
   elif pm.lda.mix_type == 'direct':
       minimizer = minimize.CGMinimizer(pm, total_energy_eigf)

    
   while convergence > pm.lda.tol and iteration <= pm.lda.max_iter:
      n_old = copy.copy(n)

      v_ks = v_ext[:]+hartree_potential(pm,n)+VXC(pm,n)
      H = update_hamiltonian(pm, H, v_ks)

      if pm.lda.mix_type == 'direct': 
          waves = minimizer.gradient_step(waves, H)
          n = electron_density(pm, waves)
          en_tot = total_energy_eigf(pm,waves, n=n)

      else:
      
          n_new,waves,energies = groundstate(pm,H) # Calculate LDA density 
          en_tot = total_energy_eigv(pm,energies, n=n)

          if pm.lda.mix_type == 'pulay':
              n = mixer.mix(n_old, n_new, energies, waves.T)
          elif pm.lda.mix_type == 'linear':
              n = (1-pm.lda.mix)*n_old + pm.lda.mix*n_new
          else:
              n = n_new

          # potential mixing
          #v_ks_old = copy.copy(v_ks)
          #if pm.lda.mix == 0:
          #   v_ks[:] = v_ext[:]+hartree_potential(pm, n)+VXC(pm, n)
          #else:
          #   v_ks[:] = (1-pm.lda.mix)*v_ks_old[:]+pm.lda.mix*(v_ext[:]+hartree_potential(pm, n)+VXC(pm, n))
          #n,waves,energies,H = groundstate(pm, v_ks) # Calculate LDA density 
      convergence = np.sum(abs(n-n_old))*pm.sys.deltax
      string = 'LDA: E = {:.12f} Ha, delta n = {:.3e}, iter = {}'.format(en_tot, convergence, iteration)
      pm.sprint(string,1,newline=True)

      iteration += 1

   iteration -= 1

   pm.sprint('',1)
   if convergence > pm.lda.tol:
       string = 'LDA: Warning: convergence not reached in {} iterations. terminating self-consistency'.format(iteration)
       pm.sprint(string,1)
   else:
       pm.sprint('LDA: reached convergence in {} iterations.'.format(iteration),0)

   pm.sprint('LDA: ground-state xc energy: %s' % EXC(pm,n),1)
   v_h = hartree_potential(pm, n)
   v_xc = VXC(pm, n)

   LDA_E = total_energy_eigf(pm, waves, n=n)
   pm.sprint('LDA: ground-state energy: {}'.format(LDA_E),1)
   
   results = rs.Results()
   results.add(v_ks[:], 'gs_lda_vks')
   results.add(v_h[:], 'gs_lda_vh')
   results.add(v_xc[:], 'gs_lda_vxc')
   results.add(n[:], 'gs_lda_den')
   results.add(LDA_E, 'gs_lda_E')

   if pm.lda.save_eig:
       results.add(waves.T,'gs_lda_eigf')
       results.add(energies,'gs_lda_eigv')

   if pm.run.save:
      results.save(pm)

   if pm.run.time_dependence == True:
      for i in range(pm.sys.NE):
         Psi[i,0,:] = waves[:,i]
      v_ks_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      v_xc_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      current = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      n_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      v_ks_t[0,:] = v_ks[:]
      n_t[0,:] = n[:]
      for i in xrange(pm.sys.grid): 
         v_ks_t[1,i] = v_ks[i]+pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax))  
         v_ext[i] += pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax)) 
      for j in range(1,pm.sys.imax): 
         string = 'LDA: evolving through real time: t = ' + str(j*pm.sys.deltat) 
         pm.sprint(string,1,newline=False)
         n_t,Psi = CrankNicolson(pm, v_ks_t,Psi,n_t,j)
         if j != pm.sys.imax-1:
            v_ks_t[j+1,:] = v_ext[:]+hartree_potential(pm, n_t[j,:])+VXC(pm, n_t[j,:])
         current[j,:] = CalculateCurrentDensity(pm, n_t,j)
         v_xc_t[j,:] = VXC(pm, n_t[j,:])

      # Output results
      results.add(v_ks_t, 'td_lda_vks')
      results.add(v_xc_t, 'td_lda_vxc')
      results.add(n_t, 'td_lda_den')
      results.add(current, 'td_lda_cur')

      if pm.run.save:
         l = ['td_lda_vks','td_lda_vxc','td_lda_den','td_lda_cur']
         results.save(pm, list=l)

      pm.sprint('',1)
   return results

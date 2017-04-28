"""Performs self-consistent LDA calculation

Uses the [adiabatic] local density approximations ([A]LDA) to calculate the
[time-dependent] electron density [and current] for a system of N electrons.

Computes approximations to VKS, VH, VXC using the LDA self-consistently. 
For ground state calculations the code outputs the LDA orbitals and energies of
the system, the ground-state charge density and the Kohn-Sham potential. 
For time dependent calculation the code also outputs the time-dependent charge
and current densities and the time-dependent Kohn-Sham potential. 

Note: Uses the LDAs developed in [Entwistle2016]_ for finite slab systems
in one dimension.

"""
from __future__ import division
from __future__ import absolute_import

import pickle
import numpy as np
import scipy as sp
import copy as copy
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla

from . import RE_Utilities
from . import results as rs
from . import mix
from . import minimize


def groundstate(pm, H):
   r"""Calculates the oribitals and ground state density for the system for a given potential

    .. math:: H \psi_{i} = E_{i} \psi_{i}
                       
   parameters
   ----------
   H: array_like
     Hamiltonian matrix (band form)

   returns
   -------
   n: array_like
     density
   eigf: array_like
     normalised orbitals, index as eigf[space_index,orbital_number]
   e: array_like
     orbital energies
   """
   
   #e,eigf = spsla.eigsh(H, k=pm.sys.grid/2, which='SA') 
   #e,eigf = spla.eigh(H)
   e,eigf = spla.eig_banded(H,True) 
   
   eigf /= np.sqrt(pm.sys.deltax)
   n = electron_density(pm, eigf)

   return n,eigf,e

def electron_density(pm, orbitals):
    r"""Compute density for given orbitals

    parameters
    ----------
    orbitals: array_like
      array of properly normalised orbitals[space-index,orital number]

    returns
    -------
    n: array_like
      electron density
    """
    occupied = orbitals[:, :pm.sys.NE]
    n = np.sum(occupied*occupied.conj(), axis=1)
    return n

def ks_potential(pm, n):
    r"""Compute Kohn-Sham potential from density

    parameters
    ----------
    n: array_like
      electron density

    returns
    -------
    v_ks: array_like
      kohn-sham potential
    """
    v_ks = pm.space.v_ext + hartree_potential(pm,n) + VXC(pm,n)
    return v_ks

def banded_to_full(H):
    r"""Convert band matrix to full matrix

    For diagonalisation, the Hamiltonian matrix may be stored as a symmetric
    band matrix with H[i,:] the ith off-diagonal.
    """
    nbnd, npt = H.shape

    H_full = np.zeros((npt,npt),dtype=np.float)
    for ioff in range(nbnd):
        d = np.arange(npt-ioff)
        H_full[d,d+ioff] = H[ioff,d]
        H_full[d+ioff,d] = H[ioff,d]

    return H_full


def hamiltonian(pm, v_KS=None, wfs=None):
    r"""Compute LDA Hamiltonian

    Computes LDA Hamiltonian from a given Kohn-Sham potential.

    parameters
    ----------
    v_KS : array_like
         KS potential
    wfs : array_like
         kohn-sham orbitals. if specified, v_KS is computed from wfs
 array_like
         Hamiltonian matrix (in banded form)
    """
    # sparse version
    #sd = pm.space.second_derivative
    #T = -0.5 * sps.diags(sd,[-2,-1,0,1,2], shape=(pm.sys.grid, pm.sys.grid), dtype=np.float, format='csr') / pm.sys.deltax**2
    ##T = -0.5 * sps.diags(sd,[-1,0,1], shape=(pm.sys.grid, pm.sys.grid), dtype=np.float, format='csr') / pm.sys.deltax**2
    #if wfs is None:
    #    V = sps.diags(v_KS, 0, shape=(pm.sys.grid, pm.sys.grid), dtype=np.float, format='csr')
    #else:
    #    V = sps.diags(ks_potential(pm, electron_density(pm,wfs)), 0, shape=(pm.sys.grid, pm.sys.grid), dtype=np.float, format='csr')
    #return (T+V).toarray()

    # banded version
    sd = pm.space.second_derivative_band
    nbnd = len(sd)
    H_new = np.zeros((nbnd, pm.sys.grid), dtype=np.float)

    for i in range(nbnd):
        H_new[i,:] = -0.5 * sd[i]

    if not(wfs is None):
        v_KS = ks_potential(pm, electron_density(pm, wfs))
    H_new[0,:] += v_KS

    return H_new
    

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

def VXC(pm , Den): 
   r"""Finite LDA approximation for the Exchange-Correlation potential

   parameters
   ----------
   Den : array_like
        density

   returns array_like
        Exchange-Correlation potential
   """
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
   NE = pm.sys.NE
   if NE > 2:
       NE = 'n'
   p = dlda[NE]

   D_xc = (p['a'] + p['b'] * n + p['c'] * n**2) * n**p['d']

   return D_xc 



def total_energy_eigv(pm, eigv, eigf=None, n=None, V_H=None, V_xc=None):
   r"""Calculates the total energy of the self-consistent LDA density                 

   Relies on knowledge of Kohn-Sham eigenvalues and the density.

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

   Uses Kohn-Sham wave functions only.

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

    With our (perhaps slightly strange definition) of H, we have
      <psi|T|psi> = psi^T T psi dx

    Note: This would be much cheaper in reciprocal space

    parameters
    ----------
    eigf: array_like
      (grid, nwf) eigen functions

    """
    sd = pm.space.second_derivative
    sd_ind = pm.space.second_derivative_indices
    T = -0.5 * sps.diags(sd, sd_ind, shape=(pm.sys.grid, pm.sys.grid), dtype=np.float, format='csr')
    #T = -0.5 * sps.diags(sd, [-1,0,1], shape=(pm.sys.grid, pm.sys.grid), dtype=np.float, format='csr') / pm.sys.deltax**2

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
      for j in range(pm.sys.grid):
         for k in range(j+1):
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
   for i in range(pm.sys.grid):
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
   pm.setup_space()
   
   v_ext = pm.space.v_ext

   # take external potential for initial guess
   H = hamiltonian(pm, v_ext)
   n_inp,waves,energies = groundstate(pm, H)
   en_tot = total_energy_eigv(pm,energies, n=n_inp)

   # need n_inp and n_out to start mixing
   H = hamiltonian(pm, ks_potential(pm, n_inp))
   n_out,waves_out,energies_out = groundstate(pm, H)


   if pm.lda.scf_type == 'pulay':
       mixer = mix.PulayMixer(pm, order=pm.lda.pulay_order, preconditioner=pm.lda.pulay_preconditioner)
   elif pm.lda.scf_type == 'cg':
       minimizer = minimize.CGMinimizer(pm, total_energy_eigf)
   elif pm.lda.scf_type == 'mixh':
       minimizer = minimize.DiagMinimizer(pm, total_energy_eigf)
       H_mix = copy.copy(H)

   iteration = 1
   converged = False
   while (not converged) and iteration <= pm.lda.max_iter:
      en_tot_old = en_tot

      if pm.lda.scf_type ==  'cg': 
          # conjugate-gradient minimization
          # start with waves, H[waves]

          waves = minimizer.step(waves, banded_to_full(H))
          n_inp = electron_density(pm, waves)

          # compute total energy at n_inp
          en_tot = total_energy_eigf(pm,waves, n=n_inp)

      elif pm.lda.scf_type == 'mixh': 
          # minimization that mixes hamiltonian directly
          # start with n_inp, H[n_inp]

          n_tmp,waves_tmp,energies_tmp = groundstate(pm,H_mix)
          H_tmp = hamiltonian(pm, ks_potential(pm, n_tmp))

          H_mix = minimizer.h_step(H_mix, H_tmp)
          n_inp,waves_inp,energies_inp = groundstate(pm,H_mix)

          # compute total energy at n_inp
          en_tot = total_energy_eigv(pm,energies_inp, n=n_inp)

      else:
          # mixing schemes
          # start with n_inp, n_out

          # compute new n_inp
          if pm.lda.scf_type == 'pulay':
              n_inp = mixer.mix(n_inp, n_out, energies_out, waves_out.T)
          elif pm.lda.scf_type == 'linear':
              n_inp = (1-pm.lda.mix)*n_inp + pm.lda.mix*n_out
          else:
              n_inp = n_out

          # potential mixing
          #v_ks_old = copy.copy(v_ks)
          #if pm.lda.mix == 0:
          #   v_ks = ks_potential(n_out)
          #else:
          #   v_ks = (1-pm.lda.mix)*v_ks_old+pm.lda.mix*ks_potential(n_out)

          # compute total energy at n_inp
          en_tot = total_energy_eigv(pm,energies_out, n=n_inp)

      # compute new ks-potential, update hamiltonian
      v_ks = ks_potential(pm, n_inp)
      H = hamiltonian(pm, v_ks)

      # compute new n_out
      # Note: in minimisation schemes (cg, hmix), n_out is only needed for
      # checking self-consistency of the density and its computation could be
      # disabled for speedup
      n_out,waves_out,energies_out = groundstate(pm,H)

      gap = energies_out[pm.sys.NE]- energies_out[pm.sys.NE-1]
      if gap < 1e-3:
          s = "LDA: Warning: small KS gap {:.3e} Ha. Convergence may be slow.".format(gap)
          pm.sprint(s)

      # compute self-consistent density error
      dn = np.sum(np.abs(n_inp-n_out))*pm.sys.deltax
      de = en_tot - en_tot_old
      converged = dn < pm.lda.tol and np.abs(de) < pm.lda.etol
      s = 'LDA: E = {:.8f} Ha, de = {:+.3e}, dn = {:.3e}, iter = {}'\
              .format(en_tot, de, dn, iteration)
      pm.sprint(s,1,newline=True)

      iteration += 1

   iteration -= 1
   pm.sprint('')

   if not converged:
       s = 'LDA: Warning: convergence not reached in {} iterations. Terminating.'.format(iteration)
       pm.sprint(s,1)
   else:
       pm.sprint('LDA: reached convergence in {} iterations.'.format(iteration),0)

   # note: for minimisation techniques (cg), we could also take the input
   # density and wave functions here (for non-converged cg, the total energy of
   # the output density + wave functions may be significantly worse).
   # However, for consistency we always choose the output quantities
   # (at self-consistency, n_inp = n_out anyway).
   n = n_out
   waves = waves_out
   energies = energies_out

   v_h = hartree_potential(pm, n)
   v_xc = VXC(pm, n)
   v_ks = v_ext + v_xc + v_h
   LDA_E = total_energy_eigf(pm, waves, n=n)
   pm.sprint('LDA: ground-state energy: {}'.format(LDA_E),1)
   
   results = rs.Results()
   results.add(n[:], 'gs_lda_den')
   results.add(v_h[:], 'gs_lda_vh')
   results.add(v_xc[:], 'gs_lda_vxc')
   results.add(v_ks[:], 'gs_lda_vks')
   results.add(LDA_E, 'gs_lda_E')

   if pm.lda.save_eig:
       results.add(waves.T,'gs_lda_eigf')
       results.add(energies,'gs_lda_eigv')

   if pm.run.save:
      results.save(pm)

   Psi = np.zeros((pm.sys.NE,pm.sys.imax,pm.sys.grid), dtype=np.complex)
   if pm.run.time_dependence == True:
      for i in range(pm.sys.NE):
         Psi[i,0,:] = waves[:,i]
      v_ks_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      v_xc_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      current = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      n_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      v_ks_t[0,:] = v_ks[:]
      n_t[0,:] = n[:]
      for i in range(pm.sys.grid): 
         v_ks_t[1,i] = v_ks[i]+pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax))  
         v_ext[i] += pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax)) 
      for j in range(1,pm.sys.imax): 
         string = 'LDA: evolving through real time: t = {}'.format(j*pm.sys.deltat) 
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

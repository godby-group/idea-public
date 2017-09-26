"""Performs self-consistent LDA calculation

Uses the [adiabatic] local density approximations ([A]LDA) to calculate the
[time-dependent] electron density [and current] for a system of N electrons.

Computes approximations to V_KS, V_H, V_xc using the LDA self-consistently. 
For ground state calculations the code outputs the LDA orbitals and energies of
the system, the ground-state charge density and the Kohn-Sham potential. 
For time dependent calculations the code also outputs the time-dependent charge
and current densities and the time-dependent Kohn-Sham potential. 

Note: Uses the LDAs developed in [Entwistle2016]_ for finite slab systems, or
the LDA developed from the HEG, in one dimension.

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

from . import RE_cython
from . import results as rs
from . import mix
from . import minimize


def groundstate(pm, H):
   r"""Calculates the oribitals and ground-state density for the system for a 
   given potential.

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


def kinetic(pm):
    r"""Compute kinetic energy operator

    parameters
    ----------
    pm : array_like
        parameters object
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
    T = np.zeros((nbnd, pm.sys.grid), dtype=np.float)

    for i in range(nbnd):
        T[i,:] = -0.5 * sd[i]

    return T


def hamiltonian(pm, v_KS=None, wfs=None):
    r"""Compute LDA Hamiltonian

    Computes LDA Hamiltonian from a given Kohn-Sham potential.

    parameters
    ----------
    v_KS : array_like
         KS potential
    wfs : array_like
         kohn-sham orbitals. if specified, v_KS is computed from wfs
    returns
    -------
    H_new: array_like
         Hamiltonian matrix (in banded form)
    """
    H_new = kinetic(pm)

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


###############################################
# These are Mike's parameters for finite LDAs #
###############################################
exc_lda = {}  # parameters for \varepsilon_{xc}(n)
exc_lda[1] = {
   'a' : -1.22015237105,
   'b' : 3.68379869508,
   'c' : -11.2544116799,
   'd' : 23.1694459076,
   'e' : -26.2993138321,
   'f' : 12.2821546751,
   'g' : 0.748763421006,
}
exc_lda[2] =  {
   'a' : -1.09745306605,
   'b' : 2.88550648624,
   'c' : -7.75624825681,
   'd' : 14.2768408722,
   'e' : -14.7954437323,
   'f' : 6.42250118148,
   'g' : 0.712989331516,
}
exc_lda[3] =  {
   'a' : -1.06385204877,
   'b' : 2.70138512697,
   'c' : -7.04311672986,
   'd' : 12.5980846646,
   'e' : -12.7282959379,
   'f' : 5.40365397672,
   'g' : 0.700801965667,
}

vxc_lda = {}  # parameters for V_{xc}(n)
for n in [1,2,3]:
    eps = exc_lda[n]
    a = eps['a']
    b = eps['b']
    c = eps['c']
    d = eps['d']
    e = eps['e']
    f = eps['f']
    g = eps['g']

    vxc_lda[n] = { 
      'a' : (g+1)*a,
      'b' : (g+2)*b,
      'c' : (g+3)*c,
      'd' : (g+4)*d,
      'e' : (g+5)*e,
      'f' : (g+6)*f,
      'g' : g,
    }

#########################################################
# These are Mike's/Michele's parameters for the HEG LDA #
#########################################################
ex_lda = {} # parameters for \varepsilon_{x}(n)
ex_lda['heg'] = {
   'a' : -1.15111280322, 
   'b' : 3.34399995728,
   'c' : -9.70787906356,
   'd' : 19.0880351582,
   'e' : -20.8961904627,
   'f' : 9.48614266406,
   'g' : 0.735861727146,
}

ec_lda = {} # parameters for \varepsilon_{c}(n)
ec_lda['heg'] = {
   'a' :  0.00166042096868,
   'b' :  0.065638899567,
   'c' :  0.0740628539892,
   'd' :  0.00406836067366,
   'e' :  0.000621193747143,
}

vx_lda = {} # parameters for V_{x}(n)
eps = ex_lda['heg']
a = eps['a']
b = eps['b']
c = eps['c']
d = eps['d']
e = eps['e']
f = eps['f']
g = eps['g']

vx_lda['heg'] = {
   'a' : (g+1)*a,
   'b' : (g+2)*b,
   'c' : (g+3)*c,
   'd' : (g+4)*d,
   'e' : (g+5)*e,
   'f' : (g+6)*f,
   'g' : g,
}


def EXC(pm, n): 
   r"""Finite/HEG LDA approximation for the Exchange-Correlation energy

   .. math ::
        E_{xc} = \int \varepsilon_{xc}(n(r)) n(r) dr

   parameters
   ----------
   n : array_like
        density

   returns float
        Exchange-Correlation energy
   """
   NE = pm.lda.NE

   if(NE != 'heg'):
       p = exc_lda[NE]
       e_xc = (p['a'] + p['b'] * n + p['c'] * n**2 + p['d'] * n**3 + p['e'] * \
              n**4 + p['f'] * n**5) * n**p['g']
   else:
       p = ex_lda[NE]
       q = ec_lda[NE]
       e_x = np.zeros(pm.sys.grid, dtype=np.float)
       e_c = np.copy(e_x)
       for j in range(pm.sys.grid):
           if(n[j] != 0.0):
               e_x[j] = (p['a'] + p['b'] * n[j] + p['c'] * n[j]**2 + p['d'] * n[j]**3 + p['e'] * \
                        n[j]**4 + p['f'] * n[j]**5) * n[j]**p['g']
 
               r_s = 0.5/n[j]
               e_c[j] = -(q['a']*r_s + q['e']*(r_s**2))/(1.0 + q['b']*r_s + \
                        q['c']*(r_s**2) + q['d']*(r_s**3))

       e_xc = e_x + e_c

   E_xc_LDA = np.dot(e_xc, n) * pm.sys.deltax

   return E_xc_LDA


def VXC(pm, Den): 
   r"""Finite/HEG LDA approximation for the exchange-correlation potential

   parameters
   ----------
   Den : array_like
        density

   returns array_like
        Exchange-Correlation potential
   """
   NE = pm.lda.NE

   if(NE != 'heg'):
       p = vxc_lda[NE]
       V_xc = (p['a'] + p['b'] * Den + p['c'] * Den**2 + p['d'] * Den**3 + \
              p['e'] * Den**4 + p['f'] * Den**5) * Den**p['g']
   else: 
       p = vx_lda[NE]
       q = ec_lda[NE]
       V_x = np.zeros(pm.sys.grid, dtype=np.float)
       V_c = np.copy(V_x)
       
       for j in range(pm.sys.grid):
           if(Den[j] != 0.0):
               V_x[j] = (p['a'] + p['b']*Den[j] + p['c']*Den[j]**2 + \
                        p['d']*Den[j]**3 + p['e']*Den[j]**4 + \
                        p['f']*Den[j]**5)*Den[j]**p['g']

               r_s = 0.5/Den[j]
               term_1 = -(q['a']*r_s + q['e']*(r_s**2))/(1.0 + q['b']*r_s + \
                        q['c']*(r_s**2) + q['d']*(r_s**3))
               term_2 = -q['a']*r_s*(q['c']*(r_s**2) + 2.0*q['d']*(r_s**3) - 1.0) - \
                        q['e']*(r_s**2)*(-q['b']*r_s + q['d']*(r_s**3) - 2.0) 
               term_3 = 2.0*r_s*((q['b']*r_s + q['c']*(r_s**2) + \
                        q['d']*(r_s**3) + 1.0)**2)
       
               V_c[j] = term_1 + (term_2/term_3)

       V_xc = V_x + V_c

   return V_xc


def total_energy_eigv(pm, eigv, eigf=None, n=None, V_H=None, V_xc=None):
   r"""Calculates the total energy of the self-consistent LDA density                 

   Relies on knowledge of Kohn-Sham eigenvalues and the density.

   .. math ::
       E = \sum_i \varepsilon_i + E_{xc}[n] - E_H[n] - \int \rho(r) V_{xc}(r)dr


   parameters
   ----------
   pm : array_like
        parameters object
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


def calculate_current_density(pm, density):
    r"""Calculates the current density of a time evolving wavefunction by 
    solving the continuity equation.

    .. math::

        \frac{\partial n}{\partial t} + \nabla \cdot j = 0

    parameters
    ----------
    pm : object
        Parameters object
    density : array_like
        2D array of the time-dependent density, indexed as       
        density[time_index,space_index]

    returns array_like
        2D array of the current density, indexed as 
        current_density[time_index,space_index]
    """
    pm.sprint('', 1, newline=True)
    current_density = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
    string = 'LDA: calculating current density'
    pm.sprint(string, 1, newline=True)
    for i in range(1, pm.sys.imax):
         string = 'LDA: t = {:.5f}'.format(i*pm.sys.deltat)
         pm.sprint(string, 1, newline=False)
         J = np.zeros(pm.sys.grid, dtype=np.float)
         J = RE_cython.continuity_eqn(pm, density[i,:], density[i-1,:])
         current_density[i,:] = J[:]
    pm.sprint('', 1, newline=True)

    return current_density


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
          s = "\nLDA: Warning: small KS gap {:.3e} Ha. Convergence may be slow.".format(gap)
          pm.sprint(s)

      # compute self-consistent density error
      dn = np.sum(np.abs(n_inp-n_out))*pm.sys.deltax
      de = en_tot - en_tot_old
      converged = dn < pm.lda.tol and np.abs(de) < pm.lda.etol
      s = 'LDA: E = {:.8f} Ha, de = {:+.3e}, dn = {:.3e}, iter = {}'\
              .format(en_tot, de, dn, iteration)
      pm.sprint(s,1,newline=False)

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
   results.add(n[:], 'gs_lda{}_den'.format(pm.lda.NE))
   results.add(v_h[:], 'gs_lda{}_vh'.format(pm.lda.NE))
   results.add(v_xc[:], 'gs_lda{}_vxc'.format(pm.lda.NE))
   results.add(v_ks[:], 'gs_lda{}_vks'.format(pm.lda.NE))
   results.add(LDA_E, 'gs_lda{}_E'.format(pm.lda.NE))

   if pm.lda.save_eig:
       results.add(waves.T,'gs_lda{}_eigf'.format(pm.lda.NE))
       results.add(energies,'gs_lda{}_eigv'.format(pm.lda.NE))

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

         # Verify orthogonality of states 
         S = np.dot(Psi[:,j,:].conj(), Psi[:,j,:].T) * pm.sys.deltax
         orthogonal = np.allclose(S, np.eye(pm.sys.NE, dtype=np.complex),atol=1e-6)
         if not orthogonal:
             pm.sprint("LDA: Warning: Orthonormality of orbitals violated at iteration {}".format(j))

         v_xc_t[j,:] = VXC(pm, n_t[j,:])

      # Calculate the current density
      current_density = calculate_current_density(pm, n_t)

      # Output results
      results.add(v_ks_t, 'td_lda{}_vks'.format(pm.lda.NE))
      results.add(v_xc_t, 'td_lda{}_vxc'.format(pm.lda.NE))
      results.add(n_t, 'td_lda{}_den'.format(pm.lda.NE))
      results.add(current_density, 'td_lda_cur')

      if pm.run.save:
         l = ['td_lda{}_vks'.format(pm.lda.NE),'td_lda{}_vxc'.format(pm.lda.NE),'td_lda{}_den'.format(pm.lda.NE),'td_lda{}_cur'.format(pm.lda.NE)]
         results.save(pm, list=l)

      pm.sprint('',1)
   return results

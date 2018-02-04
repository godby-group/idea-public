"""Performs self-consistent MLP calculation

Uses the [adiabatic] local density approximations ([A]LDA) and the 
[time-dependent] SOA to calculate the [time-dependent] electron density 
[and current] for a system of N electrons.The mixing term, f, is assumed to be 
constant. It requires the average ELF and has been optimsed for 1D systems.

Computes approximations to V_KS, V_H, V_xc using the MLP self-consistently.
For ground state calculations the code outputs the MLP orbitals, the ground-state 
charge density and the Kohn-Sham potential. For time dependent calculations the 
code also outputs the time-dependent charge and current densities and the 
time-dependent Kohn-Sham potential.

"""
from __future__ import division
from __future__ import absolute_import

import math
import pickle
import iDEA.LDA
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

    e,eigf = spla.eig_banded(H,True)

    eigf /= np.sqrt(pm.space.delta)
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


def lda_ks_potential(pm, n):
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
    v_ks_lda = pm.space.v_ext + hartree_potential(pm,n) + iDEA.LDA.VXC(pm,n)

    return v_ks_lda


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
    # banded version
    sd = pm.space.second_derivative_band
    nbnd = len(sd)
    T = np.zeros((nbnd, pm.space.npt), dtype=np.float)

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
    return np.dot(pm.space.v_int,density)*pm.space.delta


def CrankNicolson(pm, v_ks, Psi, n, t, A_ks):
    r"""Solves Crank Nicolson Equation
    """
    Mat = LHS(pm, v_ks, t-1, A_ks)
    Mat = Mat.tocsr()
    Matin =- (Mat-sps.identity(pm.space.npt, dtype='complex'))+sps.identity(pm.space.npt, dtype='complex')
    for i in range(pm.sys.NE):
        B = Matin*Psi[i,t-1,:]
        Psi[i,t,:] = spsla.spsolve(Mat,B)
        n[t,:] = 0
        for i in range(pm.sys.NE):
            n[t,:] += abs(Psi[i,t,:])**2
    return n, Psi


def LHS(pm, v_ks, j, A_ks):
    r"""Constructs the matrix A to be used in the Crank-Nicolson solution of Ax=b when evolving the wavefunction in time (Ax=b)
    """
    frac1 = 1.0/3.0
    frac2 = 1.0/24.0
    CNLHS = sps.lil_matrix((pm.space.npt,pm.space.npt), dtype='complex') # Matrix for the left hand side of the Crank Nicolson method
    for i in range(pm.space.npt):
        CNLHS[i,i] = 1.0+0.5j*pm.sys.deltat*(1.0/pm.space.delta**2+0.5*A_ks[j,i]**2+v_ks[j,i])
    for i in range(pm.space.npt-1):
       CNLHS[i,i+1] = -0.5j*pm.sys.deltat*(0.5/pm.space.delta-(frac1)*1.0j*A_ks[j,i+1]-(frac1)*1.0j*A_ks[j,i])/pm.space.delta
    for i in range(1,pm.space.npt):
       CNLHS[i,i-1] = -0.5j*pm.sys.deltat*(0.5/pm.space.delta+(frac1)*1.0j*A_ks[j,i-1]+(frac1)*1.0j*A_ks[j,i])/pm.space.delta
    for i in range(pm.space.npt-2):	
       CNLHS[i,i+2] = -0.5j*pm.sys.deltat*(1.0j*A_ks[j,i+2]+1.0j*A_ks[j,i])*(frac2)/pm.space.delta
    for i in range(2,pm.space.npt):
       CNLHS[i,i-2] = 0.5j*pm.sys.deltat*(1.0j*A_ks[j,i-2]+1.0j*A_ks[j,i])*(frac2)/pm.space.delta
    return CNLHS


def ELF(pm, den, KS, posDef=False):
    r"""Calculate the approximate ELF
    """
    grad = np.zeros((pm.sys.NE,pm.space.npt), dtype='float') # The single particle kinetic energy density terms
    for i in range(pm.sys.NE):
       grad[i,:] = np.gradient(KS[:,i], pm.space.delta) # Gradient of the density
    gradDen = np.gradient(den,pm.space.delta)
    c = np.zeros(pm.space.npt,dtype='float') # Unscaled measure
    for i in range(pm.sys.NE):
       c += np.abs(grad[i,:])**2
    c -= 0.25*((np.abs(gradDen)**2)/den)
    if posDef == True: # Force a positive-definate approximation if requested
       for i in range(den.shape[0]):
           if c[i] < 0.0:
              c[i] = 0.0
    elf = np.arange(den.shape[0]) # Scaling reference to the homogenous electron gas
    c_h = getc_h(den)
    elf = (1 + (c/c_h)**2)**(-1) # Finaly scale c to make ELF
    return elf

def getc_h(den):
    r"""Scaling term for the approximate ELF
    """
    c_h = np.arange(den.shape[0])
    c_h = (1.0/6.0)*(np.pi**2)*(den**3)
    return c_h

def extrapolate_edge(pm, A, n):
    r"""Extrapolate quantity at teh edge of the system
    """
    edge = int((5.0/100.0)*(pm.space.npt-1)) # Define the edge of the system (%)
    dAdx = np.zeros(pm.space.npt, dtype='float')
    for i in range(edge+1):
       l = edge - i
       dAdx[:] = np.gradient(A[:], pm.space.delta)
       A[l] = 8*A[l+1]-8*A[l+3]+A[l+4]+dAdx[l+2]*12.0*pm.space.delta 
    for i in range((pm.space.npt-edge),pm.space.npt):
       dAdx[:] = np.gradient(A[:], pm.space.delta)
       A[i] = 8*A[i-1]-8*A[i-3]+A[i-4]-dAdx[i-2]*12.0*pm.space.delta
    return A

def soa_ks_potential(pm, den):
    r"""Given n returns SOA potential
    """
    v_ks_soa = np.zeros(pm.space.npt, dtype='float')
    v_ks_soa = 0.25*(np.gradient(np.gradient(np.log(den),pm.space.delta),pm.space.delta))+0.125*np.gradient(np.log(den),pm.space.delta)**2 
    return v_ks_soa


def td_soa_ks_potential(pm, den, cur, j, damping2):
    r"""Given n returns TDSOA potential
    """
    v_ks_soa_t = np.zeros(pm.space.npt,dtype='float')
    v_old = np.zeros(pm.space.npt,dtype='float')
    vel = np.zeros(pm.space.npt,dtype='float')
    v_ks_soa_t = 0.25*(np.gradient(np.gradient(np.log(den[j,:]),pm.space.delta),pm.space.delta))+0.125*np.gradient(np.log(den[j,:]),pm.space.delta)**2
    vel[:] = cur[j,:]/den[j,:]
    v_ks_soa_t[:] -= 0.5*vel[:]**2
    v_ks_soa_t = filter_noise(pm, v_ks_soa_t, damping2) # Remove high frequencies from vector potential
    return v_ks_soa_t

def filter_noise(pm, A, damping):
    r"""Filters out noise in A by suppressing high-frequency terms in the Fourier transform.

    """
    A_freq = np.zeros(pm.space.npt,dtype='complex')
    # Calculate the Fourier transform of A
    A_freq = np.fft.rfft(A)

    # Apply the damping function to suppress high-frequency terms
    A_freq[:] *= damping[:]

    # Calculate the inverse Fourier transform to recover the spatial
    # representation of A with the noise filtered out
    A[:] = np.fft.irfft(A_freq, len(A)).real
    return A

def calculate_current_density(pm, density_ks, j):
    r"""Calculates the Kohn-Sham electron current density, at time t+dt, from
    the time-dependent Kohn-Sham electron density by solving the continuity
    equation.

    """
    current_density_ks = np.zeros(pm.space.npt, dtype=np.float)
    current_density_ks = RE_cython.continuity_eqn(pm, density_ks[j,:], density_ks[j-1,:])

    return current_density_ks

def remove_gauge(pm, A_ks, v_ks, v_ks_gs, j):
    r"""Removes the gauge transformation that was applied to the Kohn-Sham
    potential, so that it becomes a fully scalar quantity.
    """
    # Change gauge to calculate the full Kohn-Sham (scalar) potential
    for i in range(pm.space.npt):
        for k in range(i+1):
            v_ks[i] += (A_ks[j,k] - A_ks[j-1,k])*(pm.space.delta/pm.sys.deltat)

    # Shift the Kohn-Sham potential to match the ground-state Kohn-Sham
    # potential at the centre of the system
    shift = v_ks_gs[int((pm.space.npt-1)/2)] - v_ks[int((pm.space.npt-1)/2)]
    v_ks[:] += shift

    return v_ks[:]

# Main function
def main(parameters):
   r"""Performs MLP calculation

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
   v_ks_old = v_ext
   H = hamiltonian(pm, v_ks_old)
   n,waves,energies = groundstate(pm, H)

   if pm.mlp.reference_potential=='non':
       v_ref = v_ext
       if pm.mlp.f == 'e':
           s = 'MLP: Warning: f not optimised for v_ref = v_ext'
           pm.sprint(s,1)
       else:
           f = pm.mlp.f

   iteration = 1
   converged = False
   while (not converged) and iteration <= pm.lda.max_iter:

      # mixing scheme

      # compute the mixing term, f
      if pm.mlp.f == 'e':
          elf = ELF(pm, n, waves, False)
          av_loc = np.sum(elf[:]*n[:])*pm.space.delta/pm.sys.NE # Calculate the average localisation
          f = abs(1.49*av_loc - 0.984) # Daniele's optimsed f
      else:
          f = pm.mlp.f

      # compute new ks-potential, update hamiltonian
      v_ks_lda = lda_ks_potential(pm, n)
      v_ks_soa = soa_ks_potential(pm, n)

      if pm.mlp.reference_potential=='lda':
          v_ref = v_ks_lda

      v_ks = f*v_ks_soa + (1-f)*v_ref
      v_ks = extrapolate_edge(pm, v_ks, n) # SOA usually has noise at the edges of the system and must be extrapolated

      # potential mixing
      v_ks = (1-pm.mlp.mix)*v_ks_old+pm.mlp.mix*v_ks

      H = hamiltonian(pm, v_ks)

      # compute new n
      n_old = n
      n,waves,energies = groundstate(pm, H)

      v_ks_old = v_ks

      # compute self-consistent density error
      dn = np.sum(np.abs(n-n_old))*pm.space.delta
      converged = dn < pm.mlp.tol
      s = 'MLP: f = {:.3e}, dn = {:.3e}, iter = {}'.format(f, dn, iteration)
      pm.sprint(s,1,newline=False)

      iteration += 1

   iteration -= 1
   pm.sprint('')

   if not converged:
       s = 'MLP: Warning: convergence not reached in {} iterations. Terminating.'.format(iteration)
       pm.sprint(s,1)
   else:
       pm.sprint('MLP: reached convergence in {} iterations.'.format(iteration),0)

   v_h = hartree_potential(pm, n)
   v_xc = v_ks - v_ext - v_h

   results = rs.Results()
   results.add(n[:], 'gs_mlp_den')
   results.add(v_h[:], 'gs_mlp_vh')
   results.add(v_xc[:], 'gs_mlp_vxc')
   results.add(v_ks[:], 'gs_mlp_vks')

   if pm.run.save:
      results.save(pm)

   Psi = np.zeros((pm.sys.NE,pm.sys.imax,pm.space.npt), dtype='complex')
   if pm.run.time_dependence == True:
      for i in range(pm.sys.NE):
         Psi[i,0,:] = waves[:,i]
      v_ks_t = np.zeros((pm.sys.imax,pm.space.npt), dtype='float')
      A_ks = np.zeros((pm.sys.imax,pm.space.npt), dtype='float')
      A_old = np.zeros((pm.sys.imax,pm.space.npt), dtype='float')
      v_ks_gs = np.zeros(pm.space.npt, dtype='float')
      v_pert = np.zeros(pm.space.npt, dtype='float')
      damping = np.zeros((int(0.5*(pm.space.npt-1)+1)), dtype='float')
      damping2 = np.zeros((int(0.5*(pm.space.npt-1)+1)), dtype='float')
      v_xc_t = np.zeros((pm.sys.imax,pm.space.npt), dtype='float')
      v_h_t = np.zeros((pm.sys.imax,pm.space.npt), dtype='float')
      current = np.zeros((pm.sys.imax,pm.space.npt), dtype='float')
      n_t = np.zeros((pm.sys.imax,pm.space.npt), dtype='float')
      v_ks_t[0,:] = v_ks[:]
      v_ks_gs[:] = v_ks[:]
      v_h_t[0,:] = v_h[:]
      v_xc_t[0,:] = v_xc[:]
      n_t[0,:] = n[:]
      for i in range(pm.space.npt):
         v_pert[i] = pm.sys.v_pert((i*pm.space.delta-pm.sys.xmax))
      for i in range(int(0.5*(pm.space.npt-1)+1)):
         damping[i] = math.exp(-0.25*(i*pm.space.delta)**2) # This may need to be tuned for each system
         damping2[i] = math.exp(-0.05*(i*pm.space.delta)**2) # This may need to be tuned for each system
      v_ks_t[0,:] += v_pert[:]
      if pm.mlp.reference_potential=='non':
         v_ref = v_ext + v_pert
      for j in range(1,pm.sys.imax):
         string = 'MLP: evolving through real time: t = {}'.format(j*pm.sys.deltat)
         pm.sprint(string,1,newline=False)
         n_t, Psi = CrankNicolson(pm, v_ks_t, Psi, n_t, j, A_ks)
         current[j,:] = calculate_current_density(pm, n_t, j)
         if pm.mlp.tdf == 'a' and pm.mlp.f == 'e':
             elf = ELF(pm, n_t[j,:], waves, False)
             av_loc = np.sum(elf[:]*n_t[j,:])*pm.space.delta/pm.sys.NE # Calculate the average localisation
             f = abs(1.49*av_loc - 0.984) # Daniele's optimsed f
         A_ks[j,:] = -f*current[j,:]/n_t[j,:]
         A_ks[j,:] = filter_noise(pm, A_ks[j,:], damping)
         A_ks[j,:] = extrapolate_edge(pm, A_ks[j,:], n_t[j,:])
         if pm.mlp.reference_potential=='lda':
             v_ref = lda_ks_potential(pm, n_t[j,:]) + v_pert
         v_ks_t[j,:] = (1-f)*v_ref+f*td_soa_ks_potential(pm, n_t, current, j, damping2)
         v_ks_t[j,:] = extrapolate_edge(pm, v_ks_t[j,:], n_t[j,:])

         # Verify orthogonality of states
         S = np.dot(Psi[:,j,:].conj(), Psi[:,j,:].T) * pm.space.delta
         orthogonal = np.allclose(S, np.eye(pm.sys.NE, dtype=np.complex),atol=1e-6)
         if not orthogonal:
             pm.sprint("MLP: Warning: Orthonormality of orbitals violated at iteration {}".format(j))

      print()      
      for j in range(1,pm.sys.imax):
          st = 'MLP: transforming gauge: t = {}'.format(j*pm.sys.deltat)
          pm.sprint(st,1,newline=False)
          if pm.mlp.TDKS == True:
              # Convert vector potential into scalar potential
              v_ks_t[j,:] = remove_gauge(pm, A_ks, v_ks_t[j,:], v_ks_gs, j)
              v_h_t[j,:] = hartree_potential(pm, n_t[j,:])

      if pm.mlp.TDKS == True:
          v_ks_t[0,:] += v_ks_gs[:] - v_ks_t[0,:]
          v_xc_t[:,:] = v_ks_t[:,:] - v_ext[:] - v_pert[:] - v_h_t[:,:]
          v_xc_t[0,:] += v_pert[:]
 
      # Output results
      if pm.mlp.TDKS == True:
          results.add(v_ks_t, 'td_mlp_vks')
          results.add(v_h_t, 'td_mlp_vh')
          results.add(v_xc_t, 'td_mlp_vxc')
      results.add(n_t, 'td_mlp_den')
      results.add(current, 'td_mlp_cur')

      if pm.run.save:
          if pm.mlp.TDKS == True:
              l = ['td_mlp_vks','td_mlp_vxc','td_mlp_den','td_mlp_cur', 'td_mlp_vh']
          else:
              l = ['td_mlp_den','td_mlp_cur']
          results.save(pm, list=l)

      pm.sprint('',1)
   return results

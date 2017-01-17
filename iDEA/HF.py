"""Computes ground-state charge density of a system using the Hartree-Fock approximation. The code outputs the ground-state charge density, the 
energy of the system and the Hartree-Fock orbitals. 
"""


import copy
import pickle
import numpy as np
import scipy as sp
import scipy.linalg as spla
import results as rs


def hartree(U,density):
   r"""Constructs the hartree potential for a given density

   .. math::

       V_{H} \left( x \right) = \int_{\forall} U\left( x,x' \right) n \left( x'\right) dx'

   parameters
   ----------
   U : array_like
        Coulomb matrix
        
   density : array_like
        given density

   returns array_like
   """
   return np.dot(U,density)*dx


def coulomb():
   r"""Constructs the coulomb matrix

   .. math::

       U \left( x,x' \right) = \frac{1}{|x-x'| + 1}

   parameters
   ----------

   returns array_like
   """
   for i in range(Nx):
      xi = i*dx-0.5*L
      for j in range(Nx):
         xj = j*dx-0.5*L
         U[i,j] = 1.0/(abs(xi-xj) + pm.sys.acon)
   return U


def fock(Psi, U):
   r"""Constructs the fock operator from a set of orbitals

    .. math:: F(x,x') = \sum_{k} \psi_{k}(x) U(x,x') \psi_{k}(x')
                       

   parameters
   ----------
   Psi : array_like
        orbitals indexed as Psi[orbital_number][space_index]
   
   U : array_like
        Coulomb matrix

   returns array_like
   """
   F[:,:] = 0
   for k in range(pm.sys.NE):
      for j in range(Nx):
         for i in range(Nx):
            F[i,j] += -(np.conjugate(Psi[k,i])*U[i,j]*Psi[k,j])*dx
   return F


def groundstate(V, F):	 	
   r"""Calculates the oribitals and ground state density for the system for a given Fock operator

    .. math:: H = K + V + F \\
              H \psi_{i} = E_{i} \psi_{i}
                       

   parameters
   ----------
   V : array_like
        potential
   
   F : array_like
        Coulomb matrix

   returns array_like, array_like, array_like, array_like
        density, normalised orbitals indexed as Psi[orbital_number][space_index], potential, energies
   """					
   HGS = copy.copy(T)	
   for i in range(Nx):
      HGS[i,i] += V[i]
   if pm.hf.fock == 1:
      HGS[:,:] += F[:,:]
   K, U = spla.eigh(HGS)
   Psi = U.T / sqdx
   n_x[:] = 0
   for i in range(pm.sys.NE):
      n_x[:]+=abs(Psi[i,:])**2 
   return n_x, Psi, V, K


def main(parameters):
   r"""Performs Hartree-fock calculation

   parameters
   ----------
   parameters : object
      Parameters object

   returns object
      Results object
   """
   global verbosity, Nx, Nt, L, dx, sqdx, c, nu
   global T, n_x, n_old, n_MB, V, F, V_H, V_ext, V_add, U
   global pm
   pm = parameters

   # Import parameters
   verbosity = pm.run.verbosity
   Nx = pm.sys.grid
   Nt = pm.sys.imax
   L = 2*pm.sys.xmax
   dx = L/(Nx-1)
   sqdx = np.sqrt(dx)
   c = pm.sys.acon
   nu = pm.hf.nu
   
   # Initialise matrices
   T = np.zeros((Nx,Nx), dtype='complex')	# Kinetic energy matrix
   n_x = np.zeros(Nx)			# Charge density
   n_old = np.zeros(Nx)			# Charge density
   n_MB = np.zeros(Nx)			# Many-body charge density
   V = np.zeros(Nx)			# Matrix for the Kohn-Sham potential
   F = np.zeros((Nx,Nx),dtype='complex')   # Fock operator
   V_H = np.zeros(Nx)
   V_ext = np.zeros(Nx)
   V_add = np.zeros(Nx)
   U = np.zeros((Nx,Nx))

   # Costruct the kinetic energy matrix
   for i in range(Nx):
      for j in range(Nx):
         T[i,i] = 1.0/dx**2
         if i<Nx-1:
            T[i+1,i] = -0.5/dx**2
            T[i,i+1] = -0.5/dx**2
   
   # Construct external potential
   for i in range(Nx):
      x = i*dx-0.5*L
      V[i] = pm.sys.v_ext(x)
   V_ext[:] = V[:] 
   n_x, Psi, V, K = groundstate(V, F)
   con = 1
   
   # Construct coulomb matrix
   U = coulomb()
   
   # Calculate ground state density
   while con > pm.hf.con:
      n_old[:] = n_x[:]
      V_H = hartree(U,n_x)
      F = fock(Psi, U)
      V_add[:] = V_ext[:] + V_H[:]
      V[:] = (1-nu)*V[:] + nu*V_add[:]
      for i in range(2):		 # Smooth the edges of the system
         V[i] = V[2]
      for i in range(Nx-2,Nx):
         V[i] = V[Nx-2]
      n_x, Psi, V, K = groundstate(V, F)
      con = sum(abs(n_x[:]-n_old[:]))
      string = 'HF: computing ground-state density, convergence = ' + str(con)
      pm.sprint(string,1,newline=False)
   print
   
   # Calculate ground state energy
   E_HF = 0
   for i in range(pm.sys.NE):
      E_HF += K[i]
   for i in range(Nx):
      E_HF += -0.5*(n_x[i]*V_H[i])*dx
   for k in range(pm.sys.NE):
      for i in range(Nx):
         for j in range(Nx):
            E_HF += -0.5*(np.conjugate(Psi[k,i])*F[i,j]*Psi[k,j])*dx
   print 'HF: hartree-fock energy = %s' % E_HF.real
   
   results = rs.Results()
   results.add(E_HF.real,'gs_hf_E')
   results.add(n_x,'gs_hf_den')

   if pm.hf.save_eig:
       # Note: Psi is incorrectly normalised in the code...
       results.add(Psi, 'gs_hf_eigf')
       results.add(K, 'gs_hf_eigv')

   if pm.run.save:
      results.save(pm)
 
   return results


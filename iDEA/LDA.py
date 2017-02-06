######################################################################################
# Name: Local density approximation                                                  #
######################################################################################
# Author(s): Matt Hodgson and Mike Entwistle                                         #
######################################################################################
# Description:                                                                       #
# Computes approximations to VKS, VH, VXC using the LDA self consistently.           #
#                                                                                    #
######################################################################################
# Notes: Uses the [adiabatic] local density approximations ([A]LDA) to calculate the #
# [time-dependent] electron density [and current] for a system of N electrons.       #
#                                                                                    #
######################################################################################

import pickle
import numpy as np
import scipy as sp
import copy as copy
import RE_Utilities
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
import scipy.special as scsp
import results as rs

# Solve ground-state KS equations
def groundstate(v):
   H = copy.copy(T)
   H[0,:] += v[:]
   e,eig_func = spla.eig_banded(H,True) 
   n = np.zeros(pm.sys.grid,dtype='float')
   for i in range(pm.sys.NE):
      n[:] += abs(eig_func[:,i])**2 # Calculate density
   n[:] /= pm.sys.deltax # Normalise
   return n,eig_func,e

# Define function for generating the Hartree potential for a given charge density
def Hartree(n,U):
   return np.dot(U,n)*pm.sys.deltax

# Coulomb matrix
def Coulomb():
   U = np.zeros((pm.sys.grid,pm.sys.grid),dtype='float')
   for i in xrange(pm.sys.grid):
      for k in xrange(pm.sys.grid):	
         U[i,k] = 1.0/(abs(i*pm.sys.deltax-k*pm.sys.deltax)+pm.sys.acon)
   return U

def n_int(K, n):
   d=0.0
   for i in range(pm.sys.grid):
       d = d + (k[4] + k[5]*n[i] + k[6]*n[i]**2)*n[i]**k[7]

# LDA approximation for XC potential
def XC(Den):
   V_xc = np.zeros(pm.sys.grid,dtype='float') 
   if (pm.sys.NE == 1):
      V_xc[:] = ((-1.315+2.16*Den[:]-1.71*(Den[:])**2)*Den[:]**0.638) 
   elif (pm.sys.NE == 2):
      V_xc[:] = ((-1.19+1.77*Den[:]-1.37*(Den[:])**2)*Den[:]**0.604) 
   else:
      V_xc[:] = ((-1.24+2.1*Den[:]-1.7*(Den[:])**2)*Den[:]**0.61) 
   return V_xc

# LDA approximation for XC energy 
def EXC(Den): 
   E_xc_LDA = 0.0
   if (pm.sys.NE == 1):
      for i in xrange(pm.sys.grid):
         e_xc_LDA = ((-0.803+0.82*Den[i]-0.47*(Den[i])**2)*Den[i]**0.638) 
         E_xc_LDA += (Den[i])*(e_xc_LDA)*pm.sys.deltax
   elif (pm.sys.NE == 2):
      for i in xrange(pm.sys.grid):
         e_xc_LDA = ((-0.74+0.68*Den[i]-0.38*(Den[i])**2)*Den[i]**0.604) 
         E_xc_LDA += (Den[i])*(e_xc_LDA)*pm.sys.deltax
   else:
      for i in xrange(pm.sys.grid):
         e_xc_LDA = ((-0.77+0.79*Den[i]-0.48*(Den[i])**2)*Den[i]**0.61)
         E_xc_LDA += (Den[i])*(e_xc_LDA)*pm.sys.deltax
   return E_xc_LDA

# Function to calculate the current density
def CalculateCurrentDensity(n,j):
   J = RE_Utilities.continuity_eqn(pm.sys.grid,pm.sys.deltax,pm.sys.deltat,n[j,:],n[j-1,:])
   if pm.sys.im == 1:
      for j in xrange(pm.sys.grid):
         for k in xrange(j+1):
            x = k*pm.sys.deltax-pm.sys.xmax
            J[j] -= abs(pm.sys.v_pert_im(x))*n[j,k]*pm.sys.deltax
   return J

# Solve the Crank Nicolson equation
def CrankNicolson(v,Psi,n,j): 
   Mat = LHS(v,j)
   Mat = Mat.tocsr()
   Matin =- (Mat-sps.identity(pm.sys.grid,dtype='complex'))+sps.identity(pm.sys.grid,dtype='complex')
   for i in range(pm.sys.NE):
      B = Matin*Psi[i,j-1,:]
      Psi[i,j,:] = spsla.spsolve(Mat,B)
      n[j,:] = 0
      for i in range(pm.sys.NE):
         n[j,:] += abs(Psi[i,j,:])**2
   return n,Psi

# Left hand side of the Crank Nicolson method
def LHS(v,j):	
   CNLHS = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex') # Matrix for the left hand side of the Crank Nicholson method										
   for i in xrange(pm.sys.grid):
      CNLHS[i,i] = 1.0+0.5j*pm.sys.deltat*(1.0/pm.sys.deltax**2+v[j,i])
      if i < pm.sys.grid-1:
         CNLHS[i,i+1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax**2)
      if i > 0:
         CNLHS[i,i-1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax**2)
   return CNLHS


class PulayMixer:
    """Performs Pulay mixing

    Performs Pulay mixing with Kerker preconditioner,
    as described on p.34 of [Kresse1996]_

    """
    def __init__(self, order, pm, q0=0.79):
        """Initializes variables

        parameters
        ----------
        order: int
          order of Pulay mixing (how many densities to keep in memory)
        pm: object
          input parameters
        q0: float
          cutoff for Kerker mixing [1/a0]
          Default corresponds to 1.5/Angstrom

        """
        self.order = order
        self.step = 0
        self.x_npt = pm.sys.grid
        dtype = np.float
        self.res = np.zeros((order,self.x_npt), dtype=dtype)
        self.den_in = np.zeros((order,self.x_npt), dtype=dtype)

        self.den_delta = np.zeros((order-1,self.x_npt), dtype=dtype)
        self.res_delta = np.zeros((order-1,self.x_npt), dtype=dtype)
        self.alpha_bar = np.zeros(order-1, dtype=dtype)


        self.q0 = q0
        self.A = 1.0
        #self.G_q = np.zeros((x_npt,x_npt), dtype=np.float)
        #for i in range(x_npt):
        #    q = 2*np.pi * i/np.float(x_npt)
        #    G[i,i] = A * q**2 / (q**2 + q0**2)
        #self.G_q = np.zeros((x_npt), dtype=np.float)
        #for i in range(x_npt):
        #    q = 2*np.pi * i/np.float(x_npt)
        self.G_q = np.zeros((self.x_npt/2+1), dtype=np.float)
        dq = 2*np.pi / (2 * pm.sys.xmax)
        q0_scaled = q0 / dq
        print("scaled q {}".format(q0_scaled))
        # eqn (82)
        for i in range(len(self.G_q)):
            q = 2*np.pi * i/np.float(self.x_npt)
            self.G_q[i] = self.A * q**2 / (q**2 + q0_scaled**2)
        #self.G_r = np.set_diagonal(diagonal



    @property
    def m(self):
        r"""returns current array index m

        This corresponds to the index m usedom [Kresse1996]_ on p. 33-34
        self.mix calculates rho_in^{m+1}.
        """
        return self.step % self.order

    def update_arrays(self, den_in, den_out):
        r"""Updates densities and residuals

        Note: self.step becomes larger than self.order,
        this simply overwrites data that is no longer needed.
        """
        m = self.m

        # eqn (88)
        self.den_in[m] = den_in
        if self.step > 0:
            self.den_delta[m-1] = self.den_in[m] - self.den_in[m-1]

        self.res[m] = den_out - den_in
        if self.step > 0:
            self.res_delta[m-1] = self.res[m] - self.res[m-1]

    def compute_coefficients(self):
        r"""Computes mixing coefficients
        
        Computes

        See [Kresse1996]_ equations (87) - (90)

        .. math ::

            A_{ij} = \langle R[\rho^j_{in}] | R[\rho_{in}^i \rangle \\
            \bar{A}_{ij} = \langle \Delta R^j | \Delta R^i \rangle
        
        """

        # we return rhoin_m+1, which needs
        # * delta_dens up to m-1
        # * delta_rs up to m-1
        # * alpha_bars up to m-1
        # * rs up to m

        # * delta_dens m-1 needs rho_in_m
        # * delta_r m-1 needs r_m

        # i.e. we need delta-quantities up to m-1
        # and non-delta quantities up to m

        m = self.m

        # eqns (90,91)
        overlaps = np.dot(self.res_delta, self.res[m])
        A_bar = np.dot(self.res_delta, self.res_delta.T)
        A_bar_inv = np.linalg.inv(A_bar)
        alpha_bar = -np.dot(A_bar_inv.T,overlaps)

        return alpha_bar

    def mix(self, den_in, den_out):
        r"""Compute mix of densities
        
        See [Kresse1996]_ equation (92)
        """

        self.update_arrays(den_in, den_out)

        # until we have a sufficient history of densities, we don't mix
        if self.step < self.order:
            den_in_new = den_out
        else:
            alpha_bar = self.compute_coefficients()
            print("")
            print(alpha_bar)
            m = self.m
            # eqn (92)
            den_in_new = den_in + self.precondition(self.res[m]) \
                    + np.dot(self.alpha_bar, self.den_delta + self.precondition(self.res_delta))

        self.step = self.step + 1

        return den_in_new

    def precondition(self, f):
        """Return preconditioned f"""
        #TODO: To implement
        import matplotlib.pyplot as plt
        x = np.linspace(-10,10,self.x_npt)
        plt.plot(x,f)

        f = np.fft.rfft(f, n=self.x_npt)
        f = self.G_q * f
        f = np.fft.irfft(f, n=self.x_npt)
        plt.plot(x,f)
        plt.show()
        return f
        

# Main function
def main(parameters):
   global pm, T
   pm = parameters

   T = np.zeros((2,pm.sys.grid),dtype='float') # Kinetic Energy operator
   T[0,:] = np.ones(pm.sys.grid)/pm.sys.deltax**2 # Define kinetic energy operator							
   T[1,:] = -0.5*np.ones(pm.sys.grid)/pm.sys.deltax**2 

   if pm.lda.mix_type not in [None, 'linear', 'pulay']:
       raise ValueError("lda.mix_type must be None, 'linear' or 'pulay'")

   v_s = np.zeros(pm.sys.grid,dtype='float')
   v_ext = np.zeros(pm.sys.grid,dtype='float')
   Psi = np.zeros((pm.sys.NE,pm.sys.imax,pm.sys.grid), dtype=np.complex)
   for i in xrange(pm.sys.grid):
      v_s[i] = pm.sys.v_ext((i*pm.sys.deltax-pm.sys.xmax)) # External potential
      v_ext[i] = pm.sys.v_ext((i*pm.sys.deltax-pm.sys.xmax)) # External potential
   n,waves,energies = groundstate(v_s) #Inital guess
   U = Coulomb() # Create Coulomb matrix
   n_old = copy.deepcopy(n)
   convergence = 1.0
   iteration = 0
   #iteration = 1

   if pm.lda.mix_type == 'pulay':
       mixer = PulayMixer(order=5, pm = pm)

   while convergence > pm.lda.tol and iteration < pm.lda.max_iter:
      v_s_old = v_s
      v_s_new = v_ext[:]+Hartree(n,U)+XC(n)

      if pm.lda.mix_type == 'pulay':
          v_s = mixer.mix(v_s_old, v_s_new)
      elif pm.lda.mix_type == 'linear':
          v_s[:] = (1-pm.lda.mix)*v_s_old[:]+pm.lda.mix*v_s_new[:]
      else:
          v_s = v_s_new
      n,waves,energies = groundstate(v_s) # Calculate LDA density 
      convergence = np.sum(abs(n-n_old))*pm.sys.deltax
      n_old[:] = n[:]
      string = 'LDA: electron density convergence = {:.4e}'.format(convergence)
      pm.sprint(string,1,newline=False)
      iteration += 1
      if iteration == pm.lda.max_iter:
         string = '\nLDA: reached maximum number of iterations {}. terminating self-consistency'.format(iteration)
         pm.sprint(string,1)
         convergence = 0.0

   pm.sprint('',1)
   pm.sprint('LDA: reached convergence in {} iterations.'.format(iteration),0)
   pm.sprint('LDA: ground-state xc energy: %s' % EXC(n),1)
   v_h = Hartree(n,U)
   v_xc = XC(n)

   results = rs.Results()
   results.add(v_s[:], 'gs_lda_vks')
   results.add(v_h[:], 'gs_lda_vh')
   results.add(v_xc[:], 'gs_lda_vxc')
   results.add(n[:], 'gs_lda_den')

   if pm.lda.save_eig:
       results.add(waves.T,'gs_lda_eigf')
       results.add(energies,'gs_lda_eigv')

   if pm.run.save:
      results.save(pm)

   if pm.run.time_dependence == True:
      for i in range(pm.sys.NE):
         Psi[i,0,:] = waves[:,i]/np.sqrt(pm.sys.deltax)
      v_s_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      v_xc_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      current = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      n_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      v_s_t[0,:] = v_s[:]
      n_t[0,:] = n[:]
      for i in xrange(pm.sys.grid): 
         v_s_t[1,i] = v_s[i]+pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax))  
         v_ext[i] += pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax)) 
      for j in range(1,pm.sys.imax): 
         string = 'LDA: evolving through real time: t = ' + str(j*pm.sys.deltat) 
         pm.sprint(string,1,newline=False)
         n_t,Psi = CrankNicolson(v_s_t,Psi,n_t,j)
         if j != pm.sys.imax-1:
            v_s_t[j+1,:] = v_ext[:]+Hartree(n_t[j,:],U)+XC(n_t[j,:])
         current[j,:] = CalculateCurrentDensity(n_t,j)
         v_xc_t[j,:] = XC(n_t[j,:])

      # Output results
      results.add(v_s_t, 'td_lda_vks')
      results.add(v_xc_t, 'td_lda_vxc')
      results.add(n_t, 'td_lda_den')
      results.add(current, 'td_lda_cur')

      if pm.run.save:
         l = ['td_lda_vks','td_lda_vxc','td_lda_den','td_lda_cur']
         results.save(pm, list=l)

      pm.sprint('',1)
   return results

######################################################################################
# Name: 2 electron Exact Many Body                                                   #
######################################################################################
# Author(s): Jack Wetherell, James Ramsden                                           #
######################################################################################
# Description:                                                                       #
# Computes exact many body wavefunction and density                                  #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################

# Do not run stand-alone
if(__name__ == '__main__'):
    print('do not run stand-alone')
    quit()

# Library Imports
import mkl
import time
import copy
import pickle
import numpy as np
import scipy as sp
import RE_Utilities
import scipy.sparse as sps
import scipy.misc as spmisc
import scipy.special as spec
import scipy.sparse.linalg as spla
import create_hamiltonian_coo as coo
import results as rs

# Takes every combination of the two electron indicies and creates a single unique index
def Gind(j,k):
    return (k + j*pm.sys.grid)

# Inverses the Gind operation. Takes the single index and returns the corresponding indices used to create it.
def InvGind(jk):
    k = jk % pm.sys.grid
    j = (jk - k)/pm.sys.grid
    return j, k

# Calculates the nth Energy Eigenfunction of the Harmonic Oscillator (~H(n)(x)exp(x^2/2))
def EnergyEigenfunction(n):
    j = 0
    x = -xmax
    Psi = np.zeros(pm.sys.grid, dtype = np.cfloat)
    while (x < xmax):
        factorial = np.arange(0, n+1, 1)
        fact = np.product(factorial[1:])
        norm = (np.sqrt(1.0/((2.0**n)*fact)))*((1.0/np.pi)**0.25)
        Psi[j] = complex(norm*(spec.hermite(n)(x))*(0.25)*np.exp(-0.5*(0.25)*(x**2)), 0.0)  
        j = j + 1
        x = x + deltax
    return Psi

# Define potential array for all spacial points
def Potential(i,j,k):
    xk = -pm.sys.xmax + (k*pm.sys.deltax)
    xj = -pm.sys.xmax + (j*pm.sys.deltax)
    inte = pm.sys.interaction_strength
    if (i == 0):
        return pm.sys.v_ext(xk) + pm.sys.v_ext(xj) + inte*(1.0/(abs(xk-xj) + pm.sys.acon))
    else:
        return pm.sys.v_ext(xk) + pm.sys.v_ext(xj) + inte*(1.0/(abs(xk-xj) + pm.sys.acon)) + pm.sys.v_pert(xk) + pm.sys.v_pert(xj)


def create_hamiltonian_diagonals(i,r):
    """Create array of diagonals for the construction of H the operator.

    Evaluate the kinetic and potential values of the H operators diagonal, then
    store these in an Fortran contiguous array. This array is then passed to the
    Hamiltonian constructor create_hamiltonian_coo().

        DEPENDENT FUNCTION (external): Potential() - Used for potential evaluation.

    Args:
       i (int): Perturbation status (0 = off, 1 = on).
       r (float): Spatial location.

    Returns:
       H_diagonals (cfloat): Rank-1 array with bounds pm.sys.grid**2; Diagonals of H
       operator. The array must be Fortran contiguous.

    """
    hamiltonian_diagonals = np.zeros((pm.sys.grid**2), dtype=np.cfloat, order='F')
    const = 2.0 * pm.sys.deltax**2
    for j in range(0, pm.sys.grid):
        for k in range(0, pm.sys.grid):
            jk = Gind(j, k)
            hamiltonian_diagonals[jk] = 1.0 + (4.0*r)+ (const*r*(Potential(i, j, k)))
    return hamiltonian_diagonals


def COO_max_size(x):
    """Estimate the number of non-sparse elements in H operator.

    Return an estimate number for the total elements that exist in the
    Hamiltonian operator (banded matrix) created by create_hamiltonian_coo().
    This estimate, assuming n = spatial grid points, attempts to account for the
    diagonal (x**2), the first diagonals (2*x**2 - 4) and the sub diagonals
    (2*x**2 - 6); This will overestimate the number of elements, resulting in
    an array size larger than the total number of elements, although these are
    truncated at the point of creation thanks to the scipy.sparse.coo_matrix()
    constructor used.

    Args:
        x (float): Number of spatial grid points.

    Returns:
        Self (int): Non-sparse elements estimate in H operator.

    Raises:
        String: Warns user that more grid points are required. Returns 0.

    """
    if x<=2:
        print 'Warning: insufficient spatial grid points (Grid=>3).'
        return 0
    return int(((x**2)+(4*x**2)-10))


# Imaginary Time Crank Nicholson initial condition
def InitialconI():
    Psi1 = np.zeros(pm.sys.grid,dtype = np.cfloat)
    Psi2 = np.zeros(pm.sys.grid,dtype = np.cfloat)
    Psi1 = EnergyEigenfunction(0)
    Psi2 = EnergyEigenfunction(1)
    j = 0
    while (j < pm.sys.grid):
        k = 0
        while (k < pm.sys.grid):
            Pair = Psi1[j]*Psi2[k] - Psi1[k]*Psi2[j]
            Psiarr[0,Gind(j,k)] = Pair
            k = k + 1
        j = j + 1
    return Psiarr[0,:]

# Define function to turn array of compressed indexes into seperated indexes
def PsiConverterI(Psiarr):
    Psi2D = np.zeros((pm.sys.grid,pm.sys.grid), dtype = np.cfloat)
    mPsi2D = np.zeros((pm.sys.grid,pm.sys.grid))
    jk = 0
    while (jk < pm.sys.grid**2):
        j, k = InvGind(jk)
        Psi2D[j,k] = Psiarr[jk]
        jk = jk + 1
    mPsi2D[:,:] = (np.absolute(Psi2D[:,:])**2)
    return mPsi2D

# Define function to turn array of compressed indexes into seperated indexes
def PsiConverterR(Psiarr):
    Psi2D = np.zeros((pm.sys.grid,pm.sys.grid), dtype = np.cfloat)
    mPsi2D = np.zeros((pm.sys.grid,pm.sys.grid))
    jk = 0
    while (jk < pm.sys.grid**2):
        j, k = InvGind(jk)
        Psi2D[j,k] = Psiarr[jk]
        jk = jk + 1
    mPsi2D[:,:] = (np.absolute(Psi2D[:,:])**2)
    return mPsi2D

# Psi inverter
def PsiInverter(Psi2D,i):
    Psiarr = np.zeros((jmax**2), dtype = np.cfloat)
    j = 0
    k = 0
    while (j < jmax):
        k = 0
        while (k < kmax):
            jk = Gind(j,k)
            Psiarr[jk] = Psi2D[j,k]
            k = k + 1
        j = j + 1
    return Psiarr[:]

# Function to calulate energy of a wavefuntion
def Energy(Psi):
    a = np.linalg.norm(Psi[0,:])
    b = np.linalg.norm(Psi[1,:])
    return -(np.log(b/a))/cdeltat

# Function to construct the real matrix Af
def ConstructAf(A):
    import mkl
    A1_dat, A2_dat = mkl.mkl_split(A.data,len(A.data))
    A.data = A1_dat
    A1 = copy.copy(A)
    A.data = A2_dat
    A2 = copy.copy(A)
    Af = sps.bmat([[A1,-A2],[A2,A1]]).tocsr()
    return Af

# Function to calculate the current density
def CalculateCurrentDensity(n,i):
    J=np.zeros(pm.sys.grid)
    RE_Utilities.continuity_eqn(i+1,pm.sys.grid,pm.sys.deltax,pm.sys.deltat,n,J)
    return J

# Conctruct grid for antisym matrices
def gridz(N_x):
    N_e = 2
    Nxl = N_x**N_e
    Nxs = np.prod(range(N_x,N_x+N_e))/int(spmisc.factorial(N_e))
    sgrid = np.zeros((N_x,N_x), dtype='int')
    count = 0
    for ix in range(N_x):
        for jx in range(ix+1):
            sgrid[ix,jx] = count
            count += 1
    lgrid = np.zeros((N_x,N_x), dtype='int')
    lky = np.zeros((Nxl,2), dtype='int')
    count = 0
    for ix in range(N_x):
        for jx in range(N_x):
            lgrid[ix,jx] = count
            lky[count,0] = ix
            lky[count,1] = jx
            count += 1
    return sgrid, lgrid

# Construct antisym matrices
def antisym(N_x, retsize=False):
    N_e = 2
    Nxl = N_x**N_e
    Nxs = np.prod(range(N_x,N_x+N_e))/int(spmisc.factorial(N_e))
    sgrid, lgrid = gridz(N_x)
    C_down = sps.lil_matrix((Nxs,Nxl))
    for ix in range(N_x):
        for jx in range(ix+1):
            C_down[sgrid[ix,jx],lgrid[ix,jx]] = 1.
    C_up = sps.lil_matrix((Nxl,Nxs))
    for ix in range(N_x):
        for jx in range(N_x):
            il = lgrid[ix,jx]
            ish = sgrid[ix,jx]
            if jx <= ix:
                C_up[il,ish] = 1.
            else:
                jsh = sgrid[jx,ix]
                C_up[il,jsh] = -1.
    C_down = C_down.tocsr()
    C_up = C_up.tocsr()
    if retsize:
        return C_down, C_up, Nxs
    else:
        return C_down, C_up

# Function to calculate the current density
def calculateCurrentDensity(total_td_density):
    current_density = []
    for i in range(0,len(total_td_density)-1):
         string = 'MB: computing time dependent current density t = ' + str(i*pm.sys.deltat)
         pm.sprint(string,1,newline=False)
         J = np.zeros(pm.sys.grid)
         J = RE_Utilities.continuity_eqn(pm.sys.grid,pm.sys.deltax,pm.sys.deltat,total_td_density[i+1],total_td_density[i])
         if pm.sys.im==1:
             for j in range(pm.sys.grid):
                 for k in range(j+1):
                     x = k*pm.sys.deltax-pm.sys.xmax
                     J[j] -= abs(pm.sys.im_petrb(x))*total_td_density[i][k]*pm.sys.deltax
         current_density.append(J)
    return current_density

# Function to iterate over complex time
def CNsolveComplexTime():
    i = 1

    # Set the initial condition of the wavefunction
    Psiarr[0,:] = InitialconI()
    Psiarr_RM = c_m*Psiarr[0,:]

    # Construct array of the diagonal elements of the Hamiltonian that will be
    # passed to create_hamiltonian_coo().  The value i = 0 is passed to the
    # function ensuring no perturbation is applied (see: potential()).
    hamiltonian_diagonals = create_hamiltonian_diagonals(0, r)

    # Estimate the number of non-sparse elements that will be in the matrix form
    # of the systems hamiltonian, then initialize the sparse COOrdinate matrix
    # holding arrays with this shape.
    COO_size = COO_max_size(pm.sys.grid)
    COO_j = np.zeros((COO_size), dtype=int)
    COO_k = np.zeros((COO_size), dtype=int)
    COO_data = np.zeros((COO_size), dtype=np.cfloat)

    # Pass the holding arrays and diagonals to the hamiltonian constructor, and
    # populate the holding arrays with the coordinates and data, then convert
    # these into a sparse COOrdinate matrix.  Finally convert this into a
    # Compressed Sparse Column form for efficient arithmetic.
    #CODE CRASHES HERE
    COO_j, COO_k, COO_data = coo.create_hamiltonian_coo(COO_j, COO_k, COO_data, hamiltonian_diagonals, r, pm.sys.grid, pm.sys.grid)
    A = sps.coo_matrix((COO_data, (COO_k,COO_j)), shape=(pm.sys.grid**2, pm.sys.grid**2))
    A = sps.csc_matrix(A)

    # Construct reduction matrix of A
    A_RM = c_m * A * c_p

    # Construct the matrix C
    C = -(A-sps.identity(pm.sys.grid**2, dtype=np.cfloat))+sps.identity(pm.sys.grid**2, dtype=np.cfloat)
    C_RM = c_m*C*c_p

    # Perform iterations
    while (i < cimax):

        # Begin timing the iteration
        start = time.time()
        string = 'complex time = ' + str(i*cdeltat)
        pm.sprint(string,0)

        # Reduce the wavefunction
        if (i>=2):
            Psiarr[0,:]=Psiarr[1,:]
            Psiarr_RM = c_m*Psiarr[0,:]

        # Construct vector b
        if(par == 0):
            b_RM = C_RM*Psiarr_RM
        else:
            b_RM = mkl.mkl_mvmultiply_c(C_RM.data,C_RM.indptr+1,C_RM.indices+1,1,Psiarr_RM,C_RM.shape[0],C_RM.indices.size)

        # Solve Ax=b
        Psiarr_RM,info = spla.cg(A_RM,b_RM,x0=Psiarr_RM,tol=ctol)

        # Expand the wavefunction
        Psiarr[1,:] = c_p*Psiarr_RM

        # Calculate the energy
        Ev = Energy(Psiarr)
        string = 'energy = ' + str(Ev)
        pm.sprint(string,0)

        # Normalise the wavefunction
        mag = (np.linalg.norm(Psiarr[1,:])*deltax)
        Psiarr[1,:] = Psiarr[1,:]/mag
        
        # Stop timing the iteration
        finish = time.time()
        string = 'time to Complete Step: ' + str(finish-start)
        pm.sprint(string,0)

        # Test for convergance
        wf_con = np.linalg.norm(Psiarr[0,:]-Psiarr[1,:])
        string = 'wave function convergence: ' + str(wf_con)
        pm.sprint(string,0)
        string = 'EXT: ' + 't = ' + str(i*cdeltat) + ', convergence = ' + str(wf_con)
        pm.sprint(string,1,newline=False)
        if(i>1):
            e_con = old_energy - Ev
            string = 'energy convergence: ' + str(e_con)
            pm.sprint(string,0)
            if(e_con < ctol*10.0 and wf_con < ctol*10.0):
                print
                string = 'EXT: ground state converged' 
                pm.sprint(string,1)
                string = 'ground state converged' 
                pm.sprint(string,0)
                i = cimax
        old_energy = copy.copy(Ev)
        string = '---------------------------------------------------'
        pm.sprint(string,0)

        # Iterate
        i += 1
    
    # Dispose of matrices and terminate
    A = 0
    C = 0
    return Ev, Psiarr[1,:]

# Function to iterate over real time
def CNsolveRealTime(wavefunction):
    i = 1

    # Initialse wavefunction
    Psiarr[0,:] = wavefunction
    PsiConverterR(Psiarr[0,:])
    Psiarr_RM = c_m*Psiarr[0,:]

    # Construct array of the diagonal elements of the Hamiltonian that will
    # passed to create_hamiltonian_coo().
    hamiltonian_diagonals = create_hamiltonian_diagonals(i, r)

    # Estimate the number of non-sparse elements that will be in the matrix form
    # of the systems hamiltonian, then initialize the sparse COOrdinate matrix
    # holding arrays with this shape.
    COO_size = COO_max_size(pm.sys.grid)
    COO_j = np.zeros((COO_size), dtype=int)
    COO_k = np.zeros((COO_size), dtype=int)
    COO_data = np.zeros((COO_size), dtype=np.cfloat)

    # Pass the holding arrays and diagonals to the hamiltonian constructor, and
    # populate the holding arrays with the coordinates and data, then convert
    # these into a sparse COOrdinate matrix.  Finally convert this into a
    # Compressed Sparse Column form for efficient arithmetic.
    COO_j, COO_k, COO_data = coo.create_hamiltonian_coo(COO_j, COO_k, COO_data, hamiltonian_diagonals, r, jmax,kmax)
    A = sps.coo_matrix((COO_data, (COO_k, COO_j)), shape=(jmax**2, kmax**2))
    A = sps.csc_matrix(A)

    # Construct the reduction matrix
    A_RM = c_m*A*c_p

    # Construct the matrix Af if neccessary
    if(par == 1):
        Af = ConstructAf(A_RM)

    # Construct the matrix C
    C = -(A-sps.identity(jmax**2, dtype=np.cfloat))+sps.identity(jmax**2, dtype=np.cfloat)
    C_RM = c_m*C*c_p

    # Perform iterations
    TDD = []
    TDD_GS = []
    current_density = []

    # Save ground state
    GS = np.sum(PsiConverterR(wavefunction), axis=0)*deltax*2.0
    TDD_GS.append(GS)

    while (i <= imax):

        # Begin timing the iteration
        start = time.time()
        string = 'real time = ' + str(i*deltat) + '/' + str((imax-1)*deltat)
        pm.sprint(string,0)

        # Reduce the wavefunction
        if (i>=2):
            Psiarr[0,:] = Psiarr[1,:]
            Psiarr_RM = c_m*Psiarr[0,:]

        # Construct the vector b
        b = C*Psiarr[0,:]
        if(par == 0):
            b_RM = C_RM*Psiarr_RM
        else:
            b_RM = mkl.mkl_mvmultiply_c(C_RM.data,C_RM.indptr+1,C_RM.indices+1,1,Psiarr_RM,C_RM.shape[0],C_RM.indices.size)

        # Solve Ax=b
        if(par == 0):
            Psiarr_RM,info = spla.cg(A_RM,b_RM,x0=Psiarr_RM,tol=rtol)
        else:
            b1, b2 = mkl.mkl_split(b_RM,len(b_RM))
            bf = np.append(b1,b2)
            if(i == 1):
                xf = bf
            xf = mkl.mkl_isolve(Af.data,Af.indptr+1,Af.indices+1,1,bf,xf,Af.shape[0],Af.indices.size)
            x1, x2 = np.split(xf,2)
            Psiarr_RM = mkl.mkl_comb(x1,x2,len(x1))

        # Expand the wavefunction
        Psiarr[1,:] = c_p*Psiarr_RM

        # Convert the wavefunction
        Psi2Dcon = PsiConverterR(Psiarr[1,:])

        # Calculate denstiy
        density = np.sum(Psi2Dcon[:,:], axis=0)*deltax*2.0
        TDD.append(density)
        TDD_GS.append(density)

        # Stop timing the iteration
        finish = time.time()
        string = 'Time to Complete Step: ' + str(finish-start)
        pm.sprint(string,0)

        # Print to screen
        string = 'residual: ' + str(np.linalg.norm(A*Psiarr[1,:]-b))
        pm.sprint(string,0)
        normal = np.sum(np.absolute(Psiarr[1,:])**2)*(deltax**2)
        string = 'normalisation: ' + str(normal)
        pm.sprint(string,0)
        string = 'EXT: ' + 't = ' + str(i*deltat) + ', normalisation = ' + str(normal)
        pm.sprint(string,1,newline=False)
        string = '---------------------------------------------------'
        pm.sprint(string,0)

        # Iterate
        i += 1

    # Calculate current density
    current_density = calculateCurrentDensity(TDD_GS)

    # Dispose of matrices and terminate
    A = 0
    C = 0
    pm.sprint('',1)
    return TDD, current_density

# Call this function to run iDEA-MB for 2 electrons
def main(parameters):
        
    # Use global variables
    global jmax,kmax,xmax,tmax,deltax,deltat,imax,Psiarr,Rhv2,Psi2D,r,c_m,c_p,Nx_RM
    global cimax,cdeltat,ctol,rtol,TD,par
    global pm

    pm = parameters

    # Variable initialisation
    jmax = pm.sys.grid 
    kmax = pm.sys.grid
    xmax = pm.sys.xmax
    tmax = pm.ext.ctmax
    deltax = pm.sys.deltax
    deltat = pm.sys.deltat
    imax = pm.sys.imax
    cimax = pm.ext.cimax
    cdeltat = pm.ext.cdeltat
    ctol = pm.ext.ctol
    rtol = pm.ext.rtol
    TD = pm.run.time_dependence
    par = pm.ext.par
    verbosity = pm.run.verbosity
    c_m = 0
    c_p = 0
    Nx_RM = 0


    # Construct reduction and expansion matrices
    c_m, c_p, Nx_RM = antisym(jmax, True)

    # Complex Time array initialisations 
    string = 'EXT: constructing arrays'
    pm.sprint(string,0)
    pm.sprint(string,1)
    Psiarr = np.zeros((2,jmax**2), dtype = np.cfloat)
    Rhv2 = np.zeros((jmax**2), dtype = np.cfloat)
    Psi2D = np.zeros((jmax,kmax), dtype = np.cfloat)
    r = 0.0 + (1.0)*(cdeltat/(4.0*(deltax**2))) 

    # Evolve throught complex time
    energy, wavefunction = CNsolveComplexTime() 

    # Calculate denstiy and potential
    density = np.sum(PsiConverterI(wavefunction), axis=0)*deltax*2.0
    grid = np.linspace(-pm.sys.xmax,pm.sys.xmax,pm.sys.grid)
    potential = np.array([pm.sys.v_ext(x) for x in grid])
    
    # Save ground state density, energy and external potential
    results = rs.Results()
    results.add(density,'gs_ext_den')
    results.add(energy.real,'gs_ext_E')
    results.add(potential,'gs_ext_vxt')
    if(pm.run.save):
        results.save(pm)
        
    # Real Time array initialisations 
    if(pm.run.time_dependence == True):
        string = 'EXT: constructing arrays'
        pm.sprint(string,1)
    Psiarr = np.zeros((2,jmax**2), dtype = np.cfloat)
    Psi2D = np.zeros((jmax,kmax), dtype = np.cfloat)
    Rhv2 = np.zeros((jmax**2), dtype = np.cfloat)

    # Evolve throught real time
    if TD == True:
        tmax = pm.sys.tmax
        imax = pm.sys.imax
        deltat = tmax/(imax-1)
        deltax = pm.sys.deltax
        r = 0.0 + (1.0j)*(deltat/(4.0*(deltax**2)))
        density, current_density = CNsolveRealTime(wavefunction)
        potential = np.array([pm.sys.v_ext(x) for x in grid]) + np.array([pm.sys.v_pert(x) for x in grid])
        
        # Save time-dependent density, energy and external potential
        results.add(density,'td_ext_den')
        results.add(current_density,'td_ext_cur')
        results.add(potential,'td_ext_vxt')
        if(pm.run.save):
            l = ['td_ext_den','td_cur_e','td_ext_vxt']
            results.save(pm, list=l)
    return results

"""Calculates the exact ground-state charge density and energy for a system of two interacting electrons through solving the many-body Schrodinger equation. For a time-dependent calculation, the time-dependent charge density and current density are also outputted.
"""


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
import ELF as elf
import results as rs


def Gind(pm,j,k):
    r"""Takes every combination of the two electron indicies and creates a single unique index
    
    parameters
    ----------
    pm : object
        Parameters object
    j : integer
        1st electron index
    k : integer
        2nd electron index

    returns integer 
        Single unique index, jk
    """
    jk = k + j*pm.sys.grid

    return jk


def InvGind(pm,jk):
    r"""Inverses the Gind operation. Takes the single index and returns the two electron indices

    parameters
    ----------
    pm : object
        Parameters object
    jk : integer
        Single unique index

    returns integers
        1st electron index, j. 2nd electron index, k
    """
    k = jk % pm.sys.grid
    j = (jk - k)/pm.sys.grid

    return j,k


def EnergyEigenfunction(pm,n):
    r"""Calculates the nth energy eigenstate of the quantum harmonic oscillator

    parameters
    ----------
    pm : object
        Parameters object
    n : integer
        Principle quantum number

    returns array_like
        1D array of electron density, indexed as Psi[space_index]
    """
    Psi = np.zeros(pm.sys.grid, dtype = np.cfloat)
    factorial = np.arange(0, n+1, 1)
    fact = np.product(factorial[1:])
    norm = (np.sqrt(1.0/((2.0**n)*fact)))*((1.0/np.pi)**0.25)
    for j in range(pm.sys.grid):
        x = -pm.sys.xmax + j*pm.sys.deltax
        Psi[j] = complex(norm*(spec.hermite(n)(x))*(0.25)*np.exp(-0.5*(0.25)*(x**2)), 0.0)  

    return Psi


# Define potential array for all spacial points
def Potential(pm,i,j,k,V_ext_array,V_pert_array,V_coulomb_array):
    r"""Calculates the j,k element of the potential matrix

    parameters
    ----------
    pm : object
        Parameters object
    i : integer
        Perturbed or unperturbed system
    j : integer
        1st electron index
    k : integer
        2nd electron index
    V_ext_array : array_like
        1D array of external potential, indexed as V_ext_array[space_index] 
    V_pert_array : array_like
        1D array of perturbation, indexed as V_pert_array[space_index]
    V_coulomb_array : array_like
        1D array of Coulomb potential, indexed as V_coulomb_array[space_index]

    returns float
        j, k element of potential matrix
    """
    inte = pm.sys.interaction_strength
    if (i == 0):
        element = V_ext_array[k] + V_ext_array[j] + inte*V_coulomb_array[abs(j-k)]
    elif (i == 1):
        element = V_ext_array[k] + V_ext_array[j] + inte*V_coulomb_array[abs(j-k)] + V_pert_array[k] + V_pert_array[j]

    return element


def create_hamiltonian_diagonals(pm,i,r,V_ext_array,V_pert_array,V_coulomb_array):
    """Create array of diagonals for the construction of H the operator.

    Evaluate the kinetic and potential values of the H operators diagonal, then
    store these in an Fortran contiguous array. This array is then passed to the
    Hamiltonian constructor create_hamiltonian_coo().

        DEPENDENT FUNCTION (external): Potential() - Used for potential evaluation.

    parameters
    ----------
    pm : object
        Parameters object
    i : integer
        Perturbed or unperturbed system
    r : complex float
        Parameter in the equation Ax=b
    V_ext_array : array_like
        1D array of external potential, indexed as V_ext_array[space_index] 
    V_pert_array : array_like
        1D array of perturbation, indexed as V_pert_array[space_index]
    V_coulomb_array : array_like
        1D array of Coulomb potential, indexed as V_coulomb_array[space_index]

    returns array_like
        1D array of Hamiltonian diagonal elements, indexed as hamiltonian_diagonals[space_index] 
    """
    hamiltonian_diagonals = np.zeros((pm.sys.grid**2), dtype=np.cfloat, order='F')
    const = 2.0 * pm.sys.deltax**2
    for j in range(0, pm.sys.grid):
        for k in range(0, pm.sys.grid):
            jk = Gind(pm,j,k)
            hamiltonian_diagonals[jk] = 1.0 + (4.0*r)+ (const*r*(Potential(pm,i,j,k,V_ext_array,V_pert_array,V_coulomb_array)))

    return hamiltonian_diagonals


def COO_max_size(x):
    """Estimate the number of non-sparse elements in the Hamiltonian matrix.

    Return an estimate number for the total elements that exist in the
    Hamiltonian operator (banded matrix) created by create_hamiltonian_coo().
    This estimate, assuming n = spatial grid points, attempts to account for the
    diagonal (x**2), the first diagonals (2*x**2 - 4) and the sub diagonals
    (2*x**2 - 6); This will overestimate the number of elements, resulting in
    an array size larger than the total number of elements, although these are
    truncated at the point of creation thanks to the scipy.sparse.coo_matrix()
    constructor used.

    parameters
    ----------
    x : integer
        Number of grid points 

    returns integer
        Estimate of non-sparse elements in Hamiltonian operator
    """
    if x<=2:
        print 'Warning: insufficient spatial grid points (Grid=>3).'
        return 0

    estimate = int(((x**2)+(4*x**2)-10))

    return estimate


def InitialconI(pm,Psiarr):
    r"""Crank-Nicholson initial condition for the imaginary time propagation.

    parameters
    ----------
    pm : object
        Parameters object
    Psiarr : array_like
        2D array of wavefunction, indexed as Psiarr[time_index,space_index]

    returns array_like
        1st column of Psiarr, indexed as Psiarr[0,space_index]
    """
    Psi1 = np.zeros(pm.sys.grid,dtype = np.cfloat)
    Psi2 = np.zeros(pm.sys.grid,dtype = np.cfloat)
    Psi1 = EnergyEigenfunction(pm,0)
    Psi2 = EnergyEigenfunction(pm,1)
    j = 0
    while (j < pm.sys.grid):
        k = 0
        while (k < pm.sys.grid):
            Pair = Psi1[j]*Psi2[k] - Psi1[k]*Psi2[j]
            Psiarr[0,Gind(pm,j,k)] = Pair
            k = k + 1
        j = j + 1

    return Psiarr[0,:]


def PsiConverter(pm,wavefunction):
    r"""Turn array of compressed indices into separated indices 

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        1D array of wavefunction, indexed as wavefunction[space_index]

    returns array_like
        2D array of wavefunction, indexed as Psi2D[space_index_1,space_index_2]
    """
    Psi2D = np.zeros((pm.sys.grid,pm.sys.grid), dtype = np.cfloat)
    jk = 0
    while (jk < pm.sys.grid**2):
        j, k = InvGind(pm,jk)
        Psi2D[j,k] = wavefunction[jk]
        jk = jk + 1

    return Psi2D


def Energy(pm,Psiarr):
    r"""Calculate the energy of the system

    parameters
    ----------
    pm : object
        Parameters object
    Psiarr : array_like
        2D array of wavefunction, indexed as Psiarr[time_index,space_index]

    returns float
        Energy of the system
    """
    a = np.linalg.norm(Psiarr[0,:])
    b = np.linalg.norm(Psiarr[1,:])
    Ev = -(np.log(b/a))/pm.ext.cdeltat

    return Ev


def ConstructAf(A):
    r"""Construct the real matrix Af

    parameters
    ----------
    A : sparse matrix

    returns sparse matrix
    """
    A1_dat, A2_dat = mkl.mkl_split(A.data,len(A.data))
    A.data = A1_dat
    A1 = copy.copy(A)
    A.data = A2_dat
    A2 = copy.copy(A)
    Af = sps.bmat([[A1,-A2],[A2,A1]]).tocsr()
 
    return Af


def gridz(pm):
    r"""Construct grid for antisymmetric matrices

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like
        2D arrays of grids for antisymmetric matrices, indexed as sgrid[space_index,space_index] and lgrid[space_index,space_index]
    """
    NE = 2
    Nxl = pm.sys.grid**NE
    Nxs = np.prod(range(pm.sys.grid,pm.sys.grid+NE))/int(spmisc.factorial(NE))
    sgrid = np.zeros((pm.sys.grid,pm.sys.grid), dtype='int')
    count = 0
    for ix in range(pm.sys.grid):
        for jx in range(ix+1):
            sgrid[ix,jx] = count
            count += 1
    lgrid = np.zeros((pm.sys.grid,pm.sys.grid), dtype='int')
    lky = np.zeros((Nxl,2), dtype='int')
    count = 0
    for ix in range(pm.sys.grid):
        for jx in range(pm.sys.grid):
            lgrid[ix,jx] = count
            lky[count,0] = ix
            lky[count,1] = jx
            count += 1

    return sgrid, lgrid


def antisym(pm):
    r"""Construct antisymmetric matrices

    parameters
    ----------
    pm : object
        Parameters object

    returns sparse matrices
    """
    NE = 2
    Nxl = pm.sys.grid**NE
    Nxs = np.prod(range(pm.sys.grid,pm.sys.grid+NE))/int(spmisc.factorial(NE))
    sgrid, lgrid = gridz(pm)
    C_down = sps.lil_matrix((Nxs,Nxl))
    for ix in range(pm.sys.grid):
        for jx in range(ix+1):
            C_down[sgrid[ix,jx],lgrid[ix,jx]] = 1.
    C_up = sps.lil_matrix((Nxl,Nxs))
    for ix in range(pm.sys.grid):
        for jx in range(pm.sys.grid):
            il = lgrid[ix,jx]
            ish = sgrid[ix,jx]
            if jx <= ix:
                C_up[il,ish] = 1.
            else:
                jsh = sgrid[jx,ix]
                C_up[il,jsh] = -1.
    C_down = C_down.tocsr()
    C_up = C_up.tocsr()

    return C_down, C_up


def CalculateCurrentDensity(pm,total_td_density):
    r"""Calculate the current density

    parameters
    ----------
    pm : object
        Parameters object
    total_td_density : array_like
        2D array of time-dependent density, indexed as total_td_density[time_index,space_index]
    returns array_like
        2D array of time-dependent current density, indexed as current_density[time_index,space_index]
    """
    current_density = []
    string = 'EXT: calculating time-dependent current density'
    pm.sprint(string,1,newline=True)
    for i in range(0,len(total_td_density)-1):
         string = 'EXT: t = {:.5f}'.format((i+1)*pm.sys.deltat)
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


def CNsolveComplexTime(pm,c_m,c_p,r,Psiarr,V_ext_array,V_pert_array,V_coulomb_array):
    r"""Propagate initial wavefunction through complex time to find the ground-state of the system 

    parameters
    ----------
    pm : object
        Parameters object
    c_m : sparse matrix

    c_p : sparse matrix

    r : complex float
        Parameter in the equation Ax=b
    Psiarr : array_like
        2D array of wavefunction, indexed as Psiarr[time_index,space_index]
    V_ext_array : array_like
        1D array of external potential, indexed as V_ext_array[space_index] 
    V_pert_array : array_like
        1D array of perturbation, indexed as V_pert_array[space_index]
    V_coulomb_array : array_like
        1D array of Coulomb potential, indexed as V_coulomb_array[space_index]

    returns float and array_like
        Energy of the system. 2nd column of Psiarr, indexed as Psiarr[1,space_index]
    """
    i = 1

    # Set the initial condition of the wavefunction
    Psiarr[0,:] = InitialconI(pm,Psiarr)
    Psiarr_RM = c_m*Psiarr[0,:]

    # Construct array of the diagonal elements of the Hamiltonian that will be
    # passed to create_hamiltonian_coo().  The value i = 0 is passed to the
    # function ensuring no perturbation is applied (see: potential()).
    hamiltonian_diagonals = create_hamiltonian_diagonals(pm,0,r,V_ext_array,V_pert_array,V_coulomb_array)

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
    # compressed sparse column form for efficient arithmetic.
    COO_j, COO_k, COO_data = coo.create_hamiltonian_coo(COO_j,COO_k,COO_data,hamiltonian_diagonals,r,pm.sys.grid,pm.sys.grid)
    A = sps.coo_matrix((COO_data, (COO_k,COO_j)), shape=(pm.sys.grid**2, pm.sys.grid**2))
    A = sps.csc_matrix(A)

    # Construct reduction matrix of A
    A_RM = c_m * A * c_p

    # Construct the matrix C
    C = -(A-sps.identity(pm.sys.grid**2, dtype=np.cfloat))+sps.identity(pm.sys.grid**2, dtype=np.cfloat)
    C_RM = c_m*C*c_p

    # Perform iterations
    while (i < pm.ext.cimax):

        # Begin timing the iteration
        start = time.time()
        string = 'complex time = {:.5f}'.format(i*pm.ext.cdeltat) 
        pm.sprint(string,0,newline=True)

        # Reduce the wavefunction
        if (i>=2):
            Psiarr[0,:]=Psiarr[1,:]
            Psiarr_RM = c_m*Psiarr[0,:]

        # Construct vector b
        if(pm.ext.par == 0):
            b_RM = C_RM*Psiarr_RM
        else:
            b_RM = mkl.mkl_mvmultiply_c(C_RM.data,C_RM.indptr+1,C_RM.indices+1,1,Psiarr_RM,C_RM.shape[0],C_RM.indices.size)

        # Solve Ax=b
        Psiarr_RM,info = spla.cg(A_RM,b_RM,x0=Psiarr_RM,tol=pm.ext.ctol)

        # Expand the wavefunction
        Psiarr[1,:] = c_p*Psiarr_RM

        # Calculate the energy
        Ev = Energy(pm,Psiarr)
        string = 'energy = {:.5f}'.format(Ev)
        pm.sprint(string,0,newline=True)

        # Normalise the wavefunction
        mag = (np.linalg.norm(Psiarr[1,:])*pm.sys.deltax)
        Psiarr[1,:] = Psiarr[1,:]/mag
        
        # Stop timing the iteration
        finish = time.time()
        string = 'time to complete step: {:.5f}'.format(finish-start)
        pm.sprint(string,0,newline=True)

        # Test for convergence
        wf_con = np.linalg.norm(Psiarr[0,:]-Psiarr[1,:])
        string = 'wave function convergence: ' + str(wf_con)
        pm.sprint(string,0,newline=True) 
        if(pm.run.verbosity=='default'):
            string = 'EXT: ' + 't = {:.5f}'.format(i*pm.ext.cdeltat) + ', convergence = ' + str(wf_con)
            pm.sprint(string,1,newline=False)
        if(i>1):
            e_con = old_energy - Ev
            string = 'energy convergence: ' + str(e_con)
            pm.sprint(string,0,newline=True)
            if(e_con < pm.ext.ctol*10.0 and wf_con < pm.ext.ctol*10.0):
                i = pm.ext.cimax
                pm.sprint('',1,newline=True)
                string = 'EXT: ground-state converged' 
                pm.sprint(string,1,newline=True)
        old_energy = copy.copy(Ev)
        string = '-------------------------------------------------------------------------------------'
        pm.sprint(string,0,newline=True)

        # Iterate
        i += 1
    
    # Dispose of matrices and terminate
    A = 0
    C = 0

    return Ev, Psiarr[1,:]


def CNsolveRealTime(pm,wavefunction,c_m,c_p,r,Psiarr,V_ext_array,V_pert_array,V_coulomb_array):
    r"""Propagate ground-state wavefunction through real time 

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        1D array of the ground-state wavefunction, indexed as wavefunction[space_index]
    c_m : sparse matrix

    c_p : sparse matrix

    r : complex float
        Parameter in the equation Ax=b
    Psiarr : array_like
        2D array of wavefunction, indexed as Psiarr[time_index,space_index]
    V_ext_array : array_like
        1D array of external potential, indexed as V_ext_array[space_index] 
    V_pert_array : array_like
        1D array of perturbation, indexed as V_pert_array[space_index]
    V_coulomb_array : array_like
        1D array of Coulomb potential, indexed as V_coulomb_array[space_index]

    returns array_like and array_like
        2D array of time-dependent density, indexed as TDD[time_index,space_index]. 2D array of time-dependent current density, indexed as current_density[time_index,space_index]
    """
    i = 1

    # Initialse wavefunction
    Psiarr[0,:] = wavefunction
    Psiarr_RM = c_m*Psiarr[0,:]

    # Construct array of the diagonal elements of the Hamiltonian that will
    # passed to create_hamiltonian_coo().
    hamiltonian_diagonals = create_hamiltonian_diagonals(pm,i,r,V_ext_array,V_pert_array,V_coulomb_array)

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
    COO_j, COO_k, COO_data = coo.create_hamiltonian_coo(COO_j,COO_k,COO_data,hamiltonian_diagonals,r,pm.sys.grid,pm.sys.grid)
    A = sps.coo_matrix((COO_data, (COO_k, COO_j)), shape=(pm.sys.grid**2, pm.sys.grid**2))
    A = sps.csc_matrix(A)

    # Construct the reduction matrix
    A_RM = c_m*A*c_p

    # Construct the matrix Af if neccessary
    if(pm.ext.par == 1):
        Af = ConstructAf(A_RM)

    # Construct the matrix C
    C = -(A-sps.identity(pm.sys.grid**2, dtype=np.cfloat))+sps.identity(pm.sys.grid**2, dtype=np.cfloat)
    C_RM = c_m*C*c_p

    # Perform iterations
    TDD = []
    TDD_GS = []
    current_density = []
    ELF_TD = []

    # Save ground state
    Psi2D = PsiConverter(pm,wavefunction)
    ModPsi2D = np.absolute(Psi2D)**2
    GS = np.sum(ModPsi2D, axis=1)*pm.sys.deltax*2.0
    TDD_GS.append(GS)

    while (i <= pm.sys.imax):

        # Begin timing the iteration
        start = time.time()
        string = 'real time = {:.5f}'.format(i*pm.sys.deltat) + '/' + '{:.5f}'.format((pm.sys.imax)*pm.sys.deltat)
        pm.sprint(string,0)

        # Reduce the wavefunction
        if (i>=2):
            Psiarr[0,:] = Psiarr[1,:]
            Psiarr_RM = c_m*Psiarr[0,:]

        # Construct the vector b
        b = C*Psiarr[0,:]
        if(pm.ext.par == 0):
            b_RM = C_RM*Psiarr_RM
        else:
            b_RM = mkl.mkl_mvmultiply_c(C_RM.data,C_RM.indptr+1,C_RM.indices+1,1,Psiarr_RM,C_RM.shape[0],C_RM.indices.size)

        # Solve Ax=b
        if(pm.ext.par == 0):
            Psiarr_RM,info = spla.cg(A_RM,b_RM,x0=Psiarr_RM,tol=pm.ext.rtol)
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
        Psi2D = PsiConverter(pm,Psiarr[1,:])

        # Calculate density (and ELF)
        ModPsi2D = np.absolute(Psi2D)**2
        density = np.sum(ModPsi2D, axis=1)*pm.sys.deltax*2.0
        TDD.append(density)
        TDD_GS.append(density)
        if(pm.ext.ELF_TD == 1):
            ELF_TD.append(elf.main(pm,Psi2D))

        # Stop timing the iteration
        finish = time.time()
        string = 'time to complete step: {:.5f}'.format(finish-start)
        pm.sprint(string,0)

        # Print to screen
        string = 'residual: {:.5f}'.format(np.linalg.norm(A*Psiarr[1,:]-b))
        pm.sprint(string,0)
        normal = np.sum(np.absolute(Psiarr[1,:])**2)*(pm.sys.deltax**2)
        string = 'normalisation: {:.5f}'.format(normal)
        pm.sprint(string,0)
        if(pm.run.verbosity=='default'):
            string = 'EXT: ' + 't = {:.5f}'.format(i*pm.sys.deltat) + ', normalisation = {:.5f}'.format(normal)
            pm.sprint(string,1,newline=False)
        string = '-------------------------------------------------------------------------------------'
        pm.sprint(string,0,newline=True)

        # Iterate
        i += 1

    # Calculate current density
    pm.sprint('',1,newline=True)
    current_density = CalculateCurrentDensity(pm,TDD_GS)

    # Dispose of matrices and terminate
    A = 0
    C = 0
    pm.sprint('',1,newline=True)

    return TDD, current_density, ELF_TD


def main(parameters):
    r"""Calculates the charge density and energy (and current density) for the ground-state (time-dependent) system 

    parameters
    ----------
    parameters : object
        Parameters object

    returns object
        Results object
    """       
    pm = parameters

    # Variable initialisation
    verbosity = pm.run.verbosity
    r = 0.0 + (1.0)*(pm.ext.cdeltat/(4.0*(pm.sys.deltax**2))) 

    # Construct reduction and expansion matrices
    c_m, c_p = antisym(pm)

    # Array initialisations 
    string = 'EXT: constructing arrays'
    pm.sprint(string,1,newline=True)
    Psiarr = np.zeros((2,pm.sys.grid**2), dtype = np.cfloat)
    x_points = np.linspace(-pm.sys.xmax, pm.sys.xmax, pm.sys.grid)
    V_ext_array = pm.sys.v_ext(x_points)
    V_pert_array = pm.sys.v_pert(x_points)
    x_points_tmp = np.linspace(0.0, 2*pm.sys.xmax, pm.sys.grid)
    V_coulomb_array = 1.0/(pm.sys.acon + x_points_tmp)

    # Evolve throught complex time
    string = 'EXT: complex time evolution'
    pm.sprint(string,1,newline=True)
    energy, wavefunction = CNsolveComplexTime(pm,c_m,c_p,r,Psiarr,V_ext_array,V_pert_array,V_coulomb_array) 

    # Calculate ground-state density and external potential (and ELF)
    Psi2D = PsiConverter(pm,wavefunction)
    ModPsi2D = np.absolute(Psi2D)**2
    density = np.sum(ModPsi2D, axis=1)*pm.sys.deltax*2.0
    grid = np.linspace(-pm.sys.xmax,pm.sys.xmax,pm.sys.grid)
    potential = np.array([pm.sys.v_ext(x) for x in grid])
    if(pm.ext.ELF_GS == 1):
        ELF_GS = elf.main(pm,Psi2D)
   
    # Save ground-state density, energy and external potential (and ELF)
    results = rs.Results()
    results.add(density,'gs_ext_den')
    results.add(energy.real,'gs_ext_E')
    results.add(potential,'gs_ext_vxt')
    if(pm.ext.ELF_GS == 1):
        results.add(ELF_GS,'gs_ext_elf')
    if(pm.run.save):
        results.save(pm)
        
    # Evolve through real time
    if(pm.run.time_dependence == True):
        string = 'EXT: real time evolution'
        pm.sprint(string,1,newline=True)
        r = 0.0 + (1.0j)*(pm.sys.deltat/(4.0*(pm.sys.deltax**2)))
        density, current_density, ELF_TD = CNsolveRealTime(pm,wavefunction,c_m,c_p,r,Psiarr,V_ext_array,V_pert_array,V_coulomb_array)
        potential = np.array([pm.sys.v_ext(x) for x in grid]) + np.array([pm.sys.v_pert(x) for x in grid])
        
        # Save time-dependent density, energy and external potential (and ELF)
        results.add(density,'td_ext_den')
        results.add(current_density,'td_ext_cur')
        results.add(potential,'td_ext_vxt')
        if(pm.ext.ELF_TD == 1):
            results.add(ELF_TD,'td_ext_elf')
        if(pm.run.save):
            l = ['td_ext_den','td_ext_cur','td_ext_vxt']
            if(pm.ext.ELF_TD == 1):
                l.append('td_ext_elf')
            results.save(pm, list=l)
    return results

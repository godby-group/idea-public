"""Calculates the exact ground-state charge density and energy for a system of two interacting 
electrons through solving the many-body Schrodinger equation. If the system is perturbed,the 
time-dependent charge density and current density are calculated. The (time-dependent) ELF can 
also be calculated.
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
import ELF 
import results as rs


def gind(pm,j,k):
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


def inv_gind(pm,jk):
    r"""Inverses the gind operation. Takes the single index and returns the two electron indices

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


def energy_eigenfunction(pm,n):
    r"""Calculates the nth energy eigenstate of the quantum harmonic oscillator

    parameters
    ----------
    pm : object
        Parameters object
    n : integer
        Principle quantum number

    returns array_like
        1D array of the nth eigenstate, indexed as eigenstate[space_index]
    """
    eigenstate = np.zeros(pm.sys.grid, dtype = np.cfloat)
    factorial = np.arange(0, n+1, 1)
    fact = np.product(factorial[1:])
    norm = (np.sqrt(1.0/((2.0**n)*fact)))*((1.0/np.pi)**0.25)
    for j in range(pm.sys.grid):
        x = -pm.sys.xmax + j*pm.sys.deltax
        eigenstate[j] = complex(norm*(spec.hermite(n)(x))*(0.25)*np.exp(-0.5*(0.25)*(x**2)), 0.0)  

    return eigenstate


def potential(pm,j,k,v_ext,v_coulomb):
    r"""Calculates the j,k element of the potential matrix

    parameters
    ----------
    pm : object
        Parameters object
    j : integer
        1st electron index
    k : integer
        2nd electron index
    v_ext : array_like
        1D array of the external potential, indexed as v_ext[space_index] 
    v_coulomb : array_like
        1D array of the Coulomb potential, indexed as v_coulomb[space_index]

    returns float
        j, k element of potential matrix
    """
    inte = pm.sys.interaction_strength
    element = v_ext[k] + v_ext[j] + inte*v_coulomb[abs(j-k)]

    return element


def create_hamiltonian_diagonals(pm,r,v_ext,v_coulomb):
    """Creates an array to store the elements of the main diagonal of the Hamiltonian matrix.

    Evaluate the kinetic and potential values of the main diagonal of the Hamiltonian matrix, then 
    store these in a Fortran contiguous array. This array is then passed to the Hamiltonian        
    constructor create_hamiltonian_coo().

    parameters
    ----------
    pm : object
        Parameters object
    r : complex float
        Parameter in the equation Ax=b
    v_ext : array_like
        1D array of the external potential, indexed as v_ext[space_index] 
    v_coulomb : array_like
        1D array of the Coulomb potential, indexed as v_coulomb[space_index]

    returns array_like
        1D array of the Hamiltonian's main diagonal elements, indexed as hamiltonian_diagonals[space_index_1_2] 
    """
    hamiltonian_diagonals = np.zeros((pm.sys.grid**2), dtype=np.cfloat, order='F')
    const = 2.0 * pm.sys.deltax**2
    for j in range(pm.sys.grid):
        for k in range(pm.sys.grid):
            jk = gind(pm,j,k)
            hamiltonian_diagonals[jk] = 1.0 + 4.0*r + const*r*potential(pm,j,k,v_ext,v_coulomb)

    return hamiltonian_diagonals


def coo_max_size(pm):
    """Estimates the number of non-sparse elements in the Hamiltonian matrix.

    Returns an estimate number for the total number of elements that exist in the Hamiltonian 
    matrix (band matrix) created by create_hamiltonian_coo(). This estimate attempts to account for
    the main diagonal (x**2), the first off-diagonals (2*x**2 - 4) and the second off-diagonals
    (2*x**2 - 6). This will overestimate the number of elements, resulting in an array size larger 
    than the total number of elements, although these are truncated at the point of creation thanks
    to the scipy.sparse.coo_matrix() constructor used.

    parameters
    ----------
    pm : object
        Parameters object 

    returns integer
        Estimate of the number of non-sparse elements in the Hamiltonian matrix
    """
    if pm.sys.grid<3:
        print 'Warning: insufficient spatial grid points (Grid=>3).'
        return 0

    estimate = int(((pm.sys.grid**2)+(4*pm.sys.grid**2)-10))

    return estimate


def initial_condition(pm,wavefunction):
    r"""Initial condition for the Crank-Nicholson imaginary time propagation.

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        1D array of the wavefunction, indexed as wavefunction[space_index_1_2]

    returns array_like
        1D array of the wavefunction, indexed as wavefunction[space_index_1_2]
    """
    eigenstate_1 = energy_eigenfunction(pm,0)
    eigenstate_2 = energy_eigenfunction(pm,1)
    for j in range(pm.sys.grid):
        for k in range(pm.sys.grid):
            pair = eigenstate_1[j]*eigenstate_2[k] - eigenstate_1[k]*eigenstate_2[j]
            wavefunction[gind(pm,j,k)] = pair

    return wavefunction


def wavefunction_converter(pm,wavefunction):
    r"""Turns the array of compressed indices into separated indices 

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        1D array of the wavefunction, indexed as wavefunction[space_index_1_2]

    returns array_like
        2D array of the wavefunction, indexed as wavefunction_2D[space_index_1,space_index_2]
    """
    wavefunction_2D = np.zeros((pm.sys.grid,pm.sys.grid), dtype=np.cfloat)
    for jk in range(pm.sys.grid**2):
        j, k = inv_gind(pm,jk)
        wavefunction_2D[j,k] = wavefunction[jk]

    return wavefunction_2D


def calculate_energy(pm,wavefunction,wavefunction_old):
    r"""Calculates the energy of the system

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        1D array of the wavefunction at t, indexed as wavefunction[space_index_1_2]
    wavefunction_old : array_like
        1D array of the wavefunction at t-dt, indexed as wavefunction_old[space_index_1_2]
    returns float
        Energy of the system
    """
    a = np.linalg.norm(wavefunction_old)
    b = np.linalg.norm(wavefunction)
    energy = -(np.log(b/a))/pm.ext.cdeltat

    return energy


def construct_Af(A):
    r"""Constructs the real matrix Af

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


def construct_grids(pm):
    r"""Constructs the grids for the antisymmetric matrices

    parameters
    ----------
    pm : object
        Parameters object

    returns array_like
        2D arrays of grids for the antisymmetric matrices, indexed as sgrid[space_index,space_index] 
        and lgrid[space_index,space_index]
    """
    Nxl = pm.sys.grid**2
    Nxs = np.prod(range(pm.sys.grid,pm.sys.grid+2))/int(spmisc.factorial(2))
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


def construct_antisym(pm):
    r"""Constructs the antisymmetric matrices

    parameters
    ----------
    pm : object
        Parameters object

    returns sparse matrices
    """
    Nxl = pm.sys.grid**2
    Nxs = np.prod(range(pm.sys.grid,pm.sys.grid+2))/int(spmisc.factorial(2))
    sgrid, lgrid = construct_grids(pm)
    c_down = sps.lil_matrix((Nxs,Nxl))
    for ix in range(pm.sys.grid):
        for jx in range(ix+1):
            c_down[sgrid[ix,jx],lgrid[ix,jx]] = 1.
    c_up = sps.lil_matrix((Nxl,Nxs))
    for ix in range(pm.sys.grid):
        for jx in range(pm.sys.grid):
            il = lgrid[ix,jx]
            ish = sgrid[ix,jx]
            if jx <= ix:
                c_up[il,ish] = 1.
            else:
                jsh = sgrid[jx,ix]
                c_up[il,jsh] = -1.
    c_down = c_down.tocsr()
    c_up = c_up.tocsr()

    return c_down, c_up


def calculate_density(pm,wavefunction_2D):
    r"""Calculates the charge density

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        2D array of the wavefunction, indexed as wavefunction_2D[space_index_1,space_index_2]

    returns array_like
        1D array of the density, indexed as density[space_index]
    """
    mod_wavefunction_2D = np.absolute(wavefunction_2D)**2
    density = np.sum(mod_wavefunction_2D, axis=1)*pm.sys.deltax*2.0

    return density 


def calculate_current_density(pm,density_gs,density):
    r"""Calculates the current density

    parameters
    ----------
    pm : object
        Parameters object
    density_gs : array_like
        1D array of the ground-state density, indexed as density_gs[space_index]
    density : array_like
        2D array of the time-dependent density, indexed as density[time_index,space_index]

    returns array_like
        2D array of the current density, indexed as current_density[time_index,space_index]
    """
    pm.sprint('',1,newline=True)
    current_density = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
    string = 'EXT: calculating current density'
    pm.sprint(string,1,newline=True)
    for i in range(pm.sys.imax):
         string = 'EXT: t = {:.5f}'.format((i+1)*pm.sys.deltat)
         pm.sprint(string,1,newline=False)
         J = np.zeros(pm.sys.grid)
         if(i == 0):
             J = RE_Utilities.continuity_eqn(pm.sys.grid,pm.sys.deltax,pm.sys.deltat,density[0,:],
                                             density_gs)
         else:
             J = RE_Utilities.continuity_eqn(pm.sys.grid,pm.sys.deltax,pm.sys.deltat,density[i,:],
                                             density[i-1,:])
         if pm.sys.im==1:
             for j in range(pm.sys.grid):
                 for k in range(j+1):
                     x = k*pm.sys.deltax-pm.sys.xmax
                     if(i == 0):
                         J[j] -= abs(pm.sys.im_petrb(x))*density_gs[k]*pm.sys.deltax
                     else:
                         J[j] -= abs(pm.sys.im_petrb(x))*density[i-1,k]*pm.sys.deltax
         current_density[i,:] = J[:]
    pm.sprint('',1,newline=True)

    return current_density


def solve_imaginary_time(pm,wavefunction,c_m,c_p,v_ext,v_coulomb):
    r"""Propagates the initial wavefunction through complex time using the Crank-Nicholson method
    to find the ground-state of the system 

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        1D array of the wavefunction, indexed as wavefunction[space_index_1_2]
    c_m : sparse matrix

    c_p : sparse matrix

    v_ext : array_like
        1D array of the unperturbed external potential, indexed as v_ext[space_index] 
    v_coulomb : array_like
        1D array of the Coulomb potential, indexed as v_coulomb[space_index]

    returns float and array_like
        Energy of the system. 1D array of the ground-state wavefunction, indexed as wavefunction[space_index_1_2]
    """
    # Set the initial condition of the wavefunction
    wavefunction = initial_condition(pm,wavefunction)
    wavefunction_old = np.copy(wavefunction)

    # Construct array of the diagonal elements of the Hamiltonian that will be
    # passed to create_hamiltonian_coo().  The value i = 0 is passed to the
    # function ensuring no perturbation is applied (see: potential()).
    r = pm.ext.cdeltat/(4.0*pm.sys.deltax**2) + 0.0j
    hamiltonian_diagonals = create_hamiltonian_diagonals(pm,r,v_ext,v_coulomb)
 
    # Estimate the number of non-sparse elements that will be in the matrix form
    # of the system's Hamiltonian, then initialize the sparse COOrdinate matrix
    # holding arrays with this shape.
    coo_size = coo_max_size(pm)
    coo_j = np.zeros((coo_size), dtype=int)
    coo_k = np.zeros((coo_size), dtype=int)
    coo_data = np.zeros((coo_size), dtype=np.cfloat)

    # Pass the holding arrays and diagonals to the Hamiltonian constructor, and
    # populate the holding arrays with the coordinates and data, then convert
    # these into a sparse COOrdinate matrix.  Finally convert this into a
    # compressed sparse column form for efficient arithmetic.
    coo_j, coo_k, coo_data = coo.create_hamiltonian_coo(coo_j,coo_k,coo_data,hamiltonian_diagonals,r,pm.sys.grid,pm.sys.grid)
    A = sps.coo_matrix((coo_data,(coo_k,coo_j)), shape=(pm.sys.grid**2, pm.sys.grid**2))
    A = sps.csc_matrix(A)

    # Construct the reduction matrix of A
    A_reduced = c_m*A*c_p
 
    # Construct the matrix C
    C = -(A-sps.identity(pm.sys.grid**2, dtype=np.cfloat))+sps.identity(pm.sys.grid**2, dtype=np.cfloat)
    C_reduced = c_m*C*c_p
 
    # Print to screen
    string = 'EXT: imaginary time propagation'
    pm.sprint(string,1,newline=True)
 
    # Perform iterations
    i = 1
    while (i < pm.ext.cimax):

        # Begin timing the iteration
        start = time.time()
        string = 'complex time = {:.5f}'.format(i*pm.ext.cdeltat) 
        pm.sprint(string,0,newline=True)

        # Reduce the wavefunction
        wavefunction_reduced = c_m*wavefunction

        # Construct the vector b
        if(pm.ext.par == 0):
            b_reduced = C_reduced*wavefunction_reduced
        else:
            b_reduced = mkl.mkl_mvmultiply_c(C_reduced.data,C_reduced.indptr+1,C_reduced.indices+1,1,wavefunction_reduced,
                                             C_reduced.shape[0],C_reduced.indices.size)

        # Solve Ax=b
        wavefunction_reduced,info = spla.cg(A_reduced,b_reduced,x0=wavefunction_reduced,tol=pm.ext.ctol)

        # Expand the wavefunction
        wavefunction[:] = c_p*wavefunction_reduced

        # Calculate the energy
        energy = calculate_energy(pm,wavefunction,wavefunction_old)
        string = 'energy = {:.5f}'.format(energy)
        pm.sprint(string,0,newline=True)

        # Normalise the wavefunction
        magnitude = (np.linalg.norm(wavefunction)*pm.sys.deltax)
        wavefunction[:] = wavefunction[:]/magnitude
        
        # Stop timing the iteration
        finish = time.time()
        string = 'time to complete step: {:.5f}'.format(finish-start)
        pm.sprint(string,0,newline=True)

        # Test for convergence
        wf_con = np.linalg.norm(wavefunction_old-wavefunction)
        string = 'wave function convergence: ' + str(wf_con)
        pm.sprint(string,0,newline=True) 
        if(pm.run.verbosity=='default'):
            string = 'EXT: ' + 't = {:.5f}'.format(i*pm.ext.cdeltat) + ', convergence = ' + str(wf_con)
            pm.sprint(string,1,newline=False)
        if(i>1):
            e_con = old_energy - energy
            string = 'energy convergence: ' + str(e_con)
            pm.sprint(string,0,newline=True)
            if(e_con < pm.ext.ctol*10.0 and wf_con < pm.ext.ctol*10.0):
                i = pm.ext.cimax
                pm.sprint('',1,newline=True)
                string = 'EXT: ground-state converged' 
                pm.sprint(string,1,newline=True)
        old_energy = copy.copy(energy)
        string = '-------------------------------------------------------------------------------------'
        pm.sprint(string,0,newline=True)

        # Iterate
        wavefunction_old[:] = wavefunction[:]
        i += 1
    
    # Dispose of matrices and terminate
    A = 0
    C = 0

    return energy, wavefunction


def solve_real_time(pm,wavefunction,c_m,c_p,v_ext,v_coulomb):
    r"""Propagates the ground-state wavefunction through real time using the Crank-Nicholson method
    to find the time-evolution of the perturbed system.

    parameters
    ----------
    pm : object
        Parameters object
    wavefunction : array_like
        1D array of the ground-state wavefunction, indexed as wavefunction[space_index_1_2]
    c_m : sparse matrix

    c_p : sparse matrix

    v_ext : array_like
        1D array of the perturbed external potential, indexed as v_ext[space_index] 
    v_coulomb : array_like
        1D array of the Coulomb potential, indexed as v_coulomb[space_index]

    returns array_like and array_like
        2D array of the time-dependent density, indexed as density[time_index,space_index]. 2D array  
        of the current density, indexed as current_density[time_index,space_index]
    """
    # Construct array of the diagonal elements of the Hamiltonian that will
    # passed to create_hamiltonian_coo().
    r = 0.0 + 1.0j*pm.sys.deltat/(4.0*pm.sys.deltax**2)
    hamiltonian_diagonals = create_hamiltonian_diagonals(pm,r,v_ext,v_coulomb)

    # Estimate the number of non-sparse elements that will be in the matrix form
    # of the systems hamiltonian, then initialize the sparse COOrdinate matrix
    # holding arrays with this shape.
    coo_size = coo_max_size(pm)
    coo_j = np.zeros((coo_size), dtype=int)
    coo_k = np.zeros((coo_size), dtype=int)
    coo_data = np.zeros((coo_size), dtype=np.cfloat)

    # Pass the holding arrays and diagonals to the hamiltonian constructor, and
    # populate the holding arrays with the coordinates and data, then convert
    # these into a sparse COOrdinate matrix.  Finally convert this into a
    # Compressed Sparse Column form for efficient arithmetic.
    coo_j, coo_k, coo_data = coo.create_hamiltonian_coo(coo_j,coo_k,coo_data,hamiltonian_diagonals,r,pm.sys.grid,pm.sys.grid)
    A = sps.coo_matrix((coo_data, (coo_k, coo_j)), shape=(pm.sys.grid**2, pm.sys.grid**2))
    A = sps.csc_matrix(A)

    # Construct the reduction matrix
    A_reduced = c_m*A*c_p
 
    # Construct the matrix Af if neccessary
    if(pm.ext.par == 1):
        Af = ConstructAf(A_reduced)

    # Construct the matrix C
    C = -(A-sps.identity(pm.sys.grid**2, dtype=np.cfloat))+sps.identity(pm.sys.grid**2, dtype=np.cfloat)
    C_reduced = c_m*C*c_p

    # Array initialisations
    density = np.zeros((pm.sys.imax,pm.sys.grid), dtype=np.float)
    
    if(pm.ext.ELF_TD == 1):
        elf = np.copy(density)
    else:
        elf = 0 

    # Save ground state
    wavefunction_2D = wavefunction_converter(pm,wavefunction)
    density_gs = calculate_density(pm,wavefunction_2D)
    # Print to screen
    string = 'EXT: real time propagation'
    pm.sprint(string,1,newline=True)

    # Perform iterations
    for i in range(pm.sys.imax):

        # Begin timing the iteration
        start = time.time()
        string = 'real time = {:.5f}'.format((i+1)*pm.sys.deltat) + '/' + '{:.5f}'.format((pm.sys.imax)*pm.sys.deltat)
        pm.sprint(string,0)

        # Reduce the wavefunction
        wavefunction_reduced = c_m*wavefunction[:]

        # Construct the vector b
        b = C*wavefunction[:]
        if(pm.ext.par == 0):
            b_reduced = C_reduced*wavefunction_reduced
        else:
            b_reduced = mkl.mkl_mvmultiply_c(C_reduced.data,C_reduced.indptr+1,C_reduced.indices+1,1,wavefunction_reduced,
                                             C_reduced.shape[0],C_reduced.indices.size)

        # Solve Ax=b
        if(pm.ext.par == 0):
            wavefunction_reduced,info = spla.cg(A_reduced,b_reduced,x0=wavefunction_reduced,tol=pm.ext.rtol)
        else:
            b1, b2 = mkl.mkl_split(b_reduced,len(b_reduced))
            bf = np.append(b1,b2)
            if(i == 0):
                xf = bf
            xf = mkl.mkl_isolve(Af.data,Af.indptr+1,Af.indices+1,1,bf,xf,Af.shape[0],Af.indices.size)
            x1, x2 = np.split(xf,2)
            wavefunction_reduced = mkl.mkl_comb(x1,x2,len(x1))

        # Expand the wavefunction
        wavefunction = c_p*wavefunction_reduced

        # Calculate density (and ELF)
        wavefunction_2D = wavefunction_converter(pm,wavefunction)
        density[i,:] = calculate_density(pm,wavefunction_2D)
        if(pm.ext.ELF_TD == 1):
            elf[i,:] = ELF.main(pm,wavefunction_2D)
  
        # Stop timing the iteration
        finish = time.time()
        string = 'time to complete step: {:.5f}'.format(finish-start)
        pm.sprint(string,0)

        # Print to screen
        string = 'residual: {:.5f}'.format(np.linalg.norm(A*wavefunction-b))
        pm.sprint(string,0)
        normal = np.sum(np.absolute(wavefunction**2)*pm.sys.deltax**2)
        string = 'normalisation: {:.5f}'.format(normal)
        pm.sprint(string,0)
        if(pm.run.verbosity=='default'):
            string = 'EXT: ' + 't = {:.5f}'.format((i+1)*pm.sys.deltat)
            pm.sprint(string,1,newline=False)
        string = '-------------------------------------------------------------------------------------'
        pm.sprint(string,0,newline=True)
  
    # Calculate the current density
    current_density = calculate_current_density(pm,density_gs,density)

    # Dispose of matrices and terminate
    A = 0
    C = 0

    return density, current_density, elf


def main(parameters):
    r"""Calculates the ground-state of the system. If the system is perturbed, the time 
    evolution of the perturbed system is then calculated.

    parameters
    ----------
    parameters : object
        Parameters object

    returns object
        Results object
    """       
    pm = parameters
    
    # Construct reduction and expansion matrices
    c_m, c_p = construct_antisym(pm)

    # Array initialisations 
    string = 'EXT: constructing arrays'
    pm.sprint(string,1,newline=True)
    wavefunction = np.zeros(pm.sys.grid**2, dtype=np.cfloat)
    x_points = np.linspace(-pm.sys.xmax, pm.sys.xmax, pm.sys.grid)
    v_ext = pm.sys.v_ext(x_points)
    v_pert = pm.sys.v_pert(x_points)
    x_points_tmp = np.linspace(0.0, 2*pm.sys.xmax, pm.sys.grid)
    v_coulomb = 1.0/(pm.sys.acon + x_points_tmp)

    # Propagate throught imaginary time
    energy, wavefunction = solve_imaginary_time(pm,wavefunction,c_m,c_p,v_ext,v_coulomb) 

    # Calculate ground-state density (and ELF)
    wavefunction_2D = wavefunction_converter(pm,wavefunction)
    density = calculate_density(pm,wavefunction_2D)
    if(pm.ext.ELF_GS == 1):
        elf = ELF.main(pm,wavefunction_2D)
   
    # Save ground-state density, energy and external potential (and ELF)
    results = rs.Results()
    results.add(density,'gs_ext_den')
    results.add(energy,'gs_ext_E')
    results.add(v_ext,'gs_ext_vxt')
    if(pm.ext.ELF_GS == 1):
        results.add(elf,'gs_ext_elf')
    if(pm.run.save):
        results.save(pm)
        
    # Propagate through real time
    if(pm.run.time_dependence == True):
        string = 'EXT: constructing arrays'
        pm.sprint(string,1,newline=True)
        v_ext += v_pert
        density, current_density, elf = solve_real_time(pm,wavefunction,c_m,c_p,v_ext,v_coulomb)

        # Save time-dependent density, energy and external potential (and ELF)
        results.add(density,'td_ext_den')
        results.add(current_density,'td_ext_cur')
        results.add(v_ext,'td_ext_vxt')
        if(pm.ext.ELF_TD == 1):
            results.add(elf,'td_ext_elf')
        if(pm.run.save):
            results.save(pm)

    return results

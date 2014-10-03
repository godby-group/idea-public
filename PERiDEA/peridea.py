from math import ceil
from numpy import *
from scipy.sparse import *
from scipy.sparse.linalg import cg, spsolve
from scipy.misc import factorial
from scipy.special import binom
from scipy.linalg import eigh
import ideafort
import time
import ideafort


# ground_state_potential(): receives the system parameters and returns the 
#	user-defined ground-state external potential as a function of position.
#	Actual content of the function is up to the user.
def ground_state_potential(p):

    vbarrier = 1.0
    vgs = zeros(p.N_x)
    for ix in range(p.N_x):
        if -2. <= p.xgrid[ix] <= 2.:
            vgs[ix] = vbarrier

    return vgs


# ground_state_potential(): receives the system parameters and returns the 
#	user-defined time-dependent external potential as a function of position.
#	Actual content of the function is up to the user.
def time_dependent_potential(p):

    vgate = -0.2
    vbias = -0.2
    vbarrier = 1.
    ebias = -vbias/p.L

    vtd = zeros(p.N_x)
    for ix in range(p.N_x):
        if p.xgrid[ix] < -2.:
            vtd[ix] = ebias*p.xgrid[ix]
        elif -2. <= p.xgrid[ix] < -1.:
            vtd[ix] = vbarrier + ebias*p.xgrid[ix]
        elif -1 <= p.xgrid[ix] <= 1.:
            vtd[ix] = vgate + ebias*p.xgrid[ix]
        elif 1. < p.xgrid[ix] <= 2.:
            vtd[ix] = vbarrier + ebias*p.xgrid[ix]
        else:
            vtd[ix] = ebias*p.xgrid[ix] + vbias

    return vtd


# Main processing: Calculates the ground-state and time-dependent many-electron 
#	interacting wavefunction (and the resultant charge and current densities) 
#	yielded from the user-defined external potential.
def main():

    log("Importing parameters")

    p = get_parameters()

    if p.phase == 0.0:				# Finite boundary conditions
        aR = 0.
        aL = 0.
    else:
        aR = exp(p.phase*1.0j*pi)		# Right-hand boundary condition
        aL = aR.conjugate()			# Left-hand boundary condition

    log("Noninteracting kinetic energy calculation")

    # Fetch the second derivative finite diffence coefficients
    fdc = -0.5*finite_difference_coefficients(p.N_fdc-1, 2, p.dx)

    # Calculate the single-particle Hamiltonian matrix
    T = sp_kinetic(p, aR, fdc) + diag(ground_state_potential(p))
    e, u = eigh(T)				# Single-particle eigenstates and eigenenergies
    u /= sqrt(p.dx)				# Normalise SP wavefunctions

    Ts = 0.
    for ie in range(p.N_e):			# Total single-particle energy
        Ts += e[ie]

    # Many-body noninteracting wavefunction
    phi = slater_determinant(p, u[:,:p.N_e])	# Unreduced wavefunction
    phi = p.C_down*phi				# Reduced wavefunction

    log("Interacting Hamiltonian matrix")

    V = external_potential(p)			# Many-body external potential
    H = periodic_kinetic(p) + coulomb_operator(p) + spdiags(V, 0, p.N, p.N)	# Unreduced H
    H = p.C_down*(H*p.C_up)			# Reduced Hamiltonian
    E = p.N_perms*(p.dx**p.N_e)*sum( phi.conjugate() * (H*phi) ).real

    # Crank-Nicolson (backward and forward) time-evolution matrices
    LHS = p.C_down*( spdiags(ones(p.N), 0, p.N, p.N)*p.C_up ) + 0.5*H*p.dt
    RHS = p.C_down*( spdiags(ones(p.N), 0, p.N, p.N)*p.C_up ) - 0.5*H*p.dt

    # Calculate the initial trial energy
    E = p.N_perms*(p.dx**p.N_e)*sum( phi.conjugate() * (H*phi) ).real
    lastE = 99.*abs(E)
    t = 0
    
    log("Propagating through imaginary time")
    log("%s, %s" % (t,E))


    # Propagate as long as convergence continues
    # Propagate for at least 10 timesteps: allow things to get a bit worse to start with
    while (E < lastE) or (t < 10):

        t += 1
        lastE = E

        # Time-evolve the wavefunction
        phi, info = cg(LHS, RHS*phi, x0=phi, tol=1e-10)

        # Normalise the wavefunction
        norm = p.N_perms*(p.dx**p.N_e)*sum(abs(phi)**2)
        phi /= sqrt(norm)

        # Calculate the energy
        E = p.N_perms*(p.dx**p.N_e)*sum( phi.conjugate() * (H*phi) ).real
        log("%s %s" % (t,E))

    # Calculate and output the charge and current densities
    rho = density(p, phi)
    jay = current_density(p, phi)
    output_1dfile(p, rho, 'rho.0')
    output_1dfile(p, jay, 'jay.0')

    # Calculate the system energies

    log("Wavefunction norm: %s" % (p.N_perms*(p.dx**p.N_e)*sum(abs(phi)**2)))
    log("Noninteracting kinetic energy: %s" % (Ts.real))

    E = p.N_perms*(p.dx**p.N_e)*sum(phi.conjugate()*(H*phi)).real
    log("Total system energy: %s" % (E.real))

    H = p.C_down*(periodic_kinetic(p)*p.C_up)
    KE = p.N_perms*(p.dx**p.N_e)*sum(phi.conjugate()*(H*phi)).real
    log("Total kinetic energy: %s" % (KE.real))

    H = p.C_down*(coulomb_operator(p)*p.C_up)
    CE = p.N_perms*(p.dx**p.N_e)*sum(phi.conjugate()*(H*phi)).real
    log("Total Coulomb energy: %s" % (CE.real))

    H = 0.5*hartree_potential(p)
    E_h = p.N_perms*(p.dx**p.N_e)*sum(phi.conjugate()*(H*phi)).real
    log("Total Hartree energy: %s" % (E_h.real))

    Exc = KE.real - Ts.real + CE.real - E_h.real
    log("Exchange-correlation energy: %s" % (Exc))
    log("Exchange-correlation energy density: %s" % (Exc/p.N_e))


    log("Propagating through real time")

    V = external_potential(p, 1)		# External potential of perturbed system
    H = periodic_kinetic(p) + coulomb_operator(p) + spdiags(V, 0, p.N, p.N)	# Unreduced H
    H = p.C_down*(H*p.C_up)			# Updated reduced Hamiltonian matrix

    # Update the Crank-Nicolson operators
    LHS = p.C_down*( spdiags(ones(p.N), 0, p.N, p.N)*p.C_up ) + 0.5j*H*p.dt
    RHS = p.C_down*( spdiags(ones(p.N), 0, p.N, p.N)*p.C_up ) - 0.5j*H*p.dt

    counter = 0
    fI = open('current.out','w')		# Outputs the total current as function of time
    fN = open('CBdens.out','w')			# Outputs the total current as function of time

    # Begin real time evolution
    for t in xrange(p.N_t):

        # Time evolve one time step
        phi = spsolve(LHS, RHS*phi)
        log("%s" % (t+1))

        # Calculate the observables
        rho = density(p, phi)			# Charge density n(x,t)
        jay = current_density(p, phi)		# Current density j(x,t)
        you = jay/rho				# Velocity field u(x,t)
        Aye = p.dx*sum(jay)			# Total current I(t)
        blockade = p.dx*sum(rho[p.midx-10:p.midx+11])	# Charge in blockade
        fI.write('%s %s\n' % ((t+1)*p.dt,Aye))
        fN.write('%s %s\n' % ((t+1)*p.dt,blockade))

        # Staggered outout files (to save space)
        counter += 1
        if counter == 100:			# Can change the staggering here
            output_1dfile(p, rho, 'rho.%s' % (t+1))
            output_1dfile(p, jay, 'jay.%s' % (t+1))
            output_1dfile(p, you, 'you.%s' % (t+1))
            counter = 0

    fI.close()
    fN.close()


# Given the many-electron wavefunction, calculates the total current density
def current_density(p, phi):

    # Reshapes length N wavefunction to N_x**N_e
    psi = reshape(p.C_up*phi, p.dim)

    # Calculate the wavefunction gradients
    grads = gradient(psi)

    # Calculate the current density :- Imaginary part is zero by definition
    j = zeros(p.N_x)
    for ie in range(p.N_e):
        gradpsi = grads[ie]
        for ix in range(p.N_x):
            j[ix] += p.dx*sum((psi[ix,:].conjugate()*gradpsi[ix,:]).imag)

    return j


# Calculate the ground-state/time-dependent external potential matrix diagonal
def external_potential(p, t=0):

    if t==0.:
        vx = ground_state_potential(p)		# Gets the user-defined ground-state potential
    else:
        vx = time_dependent_potential(p)	# Gets the user-defined time-dependent potential

    output_1dfile(p, vx, 'vext.%s' % t)		# Outputs the potential

    V = zeros(p.N)				# Construct many-electron potential
    for i in xrange(p.N):
        xdim = p.xy[i,:]
        for ie in range(p.N_e):
            V[i] += vx[xdim[ie]]

    return V


# Calculates the many-electron kinetic energy matrix
def periodic_kinetic(p):

    # Fetch the finite different coefficients and their offsets from diagonal
    fdc = -0.5*finite_difference_coefficients(p.N_fdc-1, 2, p.dx)
    midf = (len(fdc)-1)/2
    offsets = (array(range(len(fdc)))-midf).tolist()

    # Calculate the kinetic energy matrix elements and their indices
    trows, tcols, tvals = ideafort.kinetic(p.N_e, p.N, p.N_fdc, p.N_x, p.N_fdc**p.N_e, offsets, p.xy+1, fdc, p.phase)

    # Construct and return the unreduced kinetic energy matrix

    return csr_matrix( (tvals,(trows,tcols)), shape=(p.N,p.N), dtype='complex')


# Calculates the finite difference coefficients for a second-derivative
def finite_difference_coefficients(N, M, dx, x0=0.):

    alpha = linspace(-dx*N/2,dx*N/2,N+1)
    delta = zeros((M+1,N+1,N+1))
    delta[0,0,0] = 1.
    c1 = 1.

    for n in range(1,N+1):
        c2 = 1.
        for v in range(n):
            c3 = alpha[n]-alpha[v]
            c2 *= c3
            if n <= M:
                delta[n,n-1,v] = 0.
            for m in range(min(n,M)+1):
                if m > 0:
                    delta[m,n,v] = ((alpha[n]-x0)*delta[m,n-1,v] - m*delta[m-1,n-1,v])/c3
                else:
                    delta[m,n,v] = (alpha[n]-x0)*delta[m,n-1,v]/c3
        for m in range(min(n,M)+1):
            if m > 0:
                delta[m,n,n] = (c1/c2)*(m*delta[m-1,n-1,n-1] - (alpha[n-1]-x0)*delta[m,n-1,n-1])
            else:
                delta[m,n,n] = -(c1/c2)*(alpha[n-1]-x0)*delta[m,n-1,n-1]
        c1 = c2

    return delta[-1,-1,:]


# Calculates the single-particle kinetic energy matrix
def sp_kinetic(p, aR, fdc):

    midf = (len(fdc)-1)/2

    # FD coef corresponding to matrix diagonal
    diags = [fdc[midf]*ones(p.N_x, dtype='complex')]
    offsets = [0]

    # Off-diagonal finite difference coefficients
    for i in range(1,midf+1):
        diags.append(fdc[midf+i]*ones(p.N_x, dtype='complex'))
        offsets.append(i)
        diags.append(fdc[midf+i]*ones(p.N_x, dtype='complex'))
        offsets.append(-i)
        diags.append(aR*fdc[midf+i]*ones(p.N_x, dtype='complex'))
        offsets.append(p.N_x-i)
        diags.append(aR.conjugate()*fdc[midf+i]*ones(p.N_x, dtype='complex'))
        offsets.append(-p.N_x+i)

    # Construct and return the single-particle sparse KE matrix

    return spdiags(diags,offsets,p.N_x,p.N_x).todense()


# Calculates the many-electron Coulomb operator matrix using iDEAfort library
def coulomb_operator(p):
    log("-- Coulomb operator")
    w = ideafort.coulomb(p.N_e, p.N, p.N_x, p.N_cells, p.xy+1, p.dx, p.C) + 0.0j
    return spdiags(p.W0*w, 0, p.N, p.N, format='lil')


# Calculates the Hartree potential matrix using iDEAfort library
def hartree_potential(p, factor=1, use_primary=1):
    hdiag = ideafort.hartree(p.N_e, p.N_x, p.N, p.N_cells, p.dx, p.L, p.C, p.xy+1, use_primary)
    return p.C_down*((p.W0/factor)*spdiags(hdiag, 0, p.N, p.N, 'lil')*p.C_up)


# Calculate the noninteracting Slater determinant
# Currently tested for two electrons only
def slater_determinant(p, u):

    phi = zeros(p.N, dtype='complex')

    # Get all indices of the wavefunction

    for i in xrange(p.N):

        # Get every permutation of the index
        perms, swaps = ideafort.permutations(p.xy[i,:], p.N_perms, p.N_e)

        # Add product wavefunction for each permutation
        for iperm in range(p.N_perms):

            xind = perms[iperm,:]
            dphi = 1. + 0.0j
            for ie in range(p.N_e):
                dphi *= u[xind[ie],ie]

            phi[i] += ((-1.)**swaps[iperm])*dphi/sqrt(1.0*p.N_perms)

#        x1, x2 = p.xy[i,0], p.xy[i,1]
#        phi[i] = ( u[x1,0]*u[x2,1] - u[x1,1]*u[x2,0] )/sqrt(1.0*p.N_e)

    return phi


# Calculate the charge density of many-electron wavefunction using iDEAfort
def density(p, phi):
    return ideafort.density(len(phi), p.N_x, p.N, p.N_e, p.xy+1, p.dx, phi)


# Output messages
def log(string):

    print string
    f = open('runlog.out','a')
    f.write(string+'\n')
    f.close()


# Mapping from 1D wavefunction index to ie^th electron coordinate
def xyi(N_e, N_x, N, ie):
    return ideafort.xygrid(N_e,N_x,N,ie)


# Outputs 1D data (e.g. charge density, current density, etc.)
def output_1dfile(p, data, filename):

    f = open(filename, 'w')
    for ix in range(p.N_x):
        f.write('%s %s %s\n' % (p.xgrid[ix], data[ix].real, data[ix].imag))
    f.write('%s %s %s\n' % (p.xgrid[0]+p.L, data[0].real, data[0].imag))
    f.close()


# Constructs the 'Parameters' class containing all generally required information
def get_parameters():

    class parameters:

        log("-- Independent parameters")

        f = open('peridea.in')

        N_e = int(f.readline().split()[0])		# Number of electrons per supercell
        L = float(f.readline().split()[0])		# Length of periodic supercell
        N_x = int(f.readline().split()[0])		# Number of grid points sampled in supercell (per electron)
        W0 = float(f.readline().split()[0])		# Strength of electron-electron interaction
        C = float(f.readline().split()[0])		# Interaction softening parameter
        N_cells = int(f.readline().split()[0])		# Number of supercells included in interaction
        phase = float(f.readline().split()[0])		# Boundary conditions
        N_cores = int(f.readline().split()[0])		# Number of cores used in calculation
        N_fdc = int(f.readline().split()[0])		# Number of finite difference coefficients in derivatives
        trial = int(f.readline().split()[0])		# Trial wavefunction:	0 = noninteracting solution,
							#			1 = random distribution
        C_t = float(f.readline().split()[0])		# dt = C_t*(dx**2)
        tdflag = int(f.readline().split()[0])		# Time-dependence:	0 = ground state only
							#			1 = time-dependent
	N_t = int(f.readline().split()[0])		# Number of time steps
        dt = float(f.readline().split()[0])		# Time step


        f.close()

        log("-- Dependent parameters")

        N = N_x**N_e
        dx = L/N_x
        dim = ()
        for ie in range(N_e):
            dim += (N_x,)
        if W0 == 0.:
            N_cells = 1
        dt = C_t*(dx**2)
        Nxs = int(round(binom(N_x,N_e)))
        N_perms = int(factorial(N_e))	
        midx = N_x/2


        log("-- Grids")

        xgrid = linspace(-0.5*L,0.5*L-dx,N_x)	# Single-particle grid
        xcoords = array(xrange(N)).reshape(dim)	# Map from single-particle grid to N-electron grid

        xy = zeros((N,N_e), dtype='int')	# Map from N-electron grid to single-particle grids
        for ie in range(N_e):
            xy[:,ie] = xyi(N_e,N_x,N,ie+1)

        sgrid = ideafort.downgrid(N_e,N,xy+1).reshape(dim)

        log("-- Reduction & expansion matrices")

#   Worryingly, this old routine (used currently in iDea) returns different results to new routine below
#   Looking at the indices (e.g. 1,2,3) and signs (e.g. -1), it appears the new is correct, the old is wrong
#    C_down_old, C_up_old = antisym(N_e, N_x, dim, xy, xcoords, sgrid, job_server)

        C_down_rows, C_down_cols, C_down_vals, C_up_rows, C_up_cols, C_up_vals = ideafort.antisym(N_e, Nxs, N, N_x, N_perms, xy+1, sgrid.reshape(N)+1)
        C_down = coo_matrix( (C_down_vals,(C_down_rows,C_down_cols)), shape=(Nxs,N)).tocsr()
        C_up = coo_matrix( (C_up_vals,(C_up_rows,C_up_cols)), shape=(N,Nxs)).tocsr()

    return parameters()


# Execute main program if script is called directly
if __name__ == "__main__":
    log("Peridea v1.2")
    main()


### Library imports
from __future__ import division
from iDEA.input import InputSection, SystemSection
import numpy as np


### Run parameters
run = InputSection()
run.name = 'run_name'                #: Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\)
run.time_dependence = True          #: whether to run time-dependent calculation
run.verbosity = 'default'            #: output verbosity ('low', 'default', 'high')
run.save = True                      #: whether to save results to disk when they are generated
run.module = 'iDEA'                  #: specify alternative folder (in this directory) containing modified iDEA module
run.NON = False                       #: Run Non-Interacting approximation
run.LDA = False                      #: Run LDA approximation
run.MLP = False                      #: Run MLP approximation
run.HF = True                       #: Run Hartree-Fock approximation
run.EXT = False                       #: Run Exact Many-Body calculation
run.HYB = False                      #: Run Hybrid (HF-LDA) calculation
run.MBPT = False                     #: Run Many-body pertubation theory
run.LAN = False                      #: Run Landauer approximation


### System parameters
sys = SystemSection()
sys.NE = 2                           #: Number of electrons
sys.grid = 201                       #: Number of grid points (must be odd)
sys.stencil = 3                      #: Discretisation of 2nd derivative (3 or 5 or 7)
sys.xmax = 10.0                      #: Size of the system
sys.tmax = 0.5                       #: Total real time
sys.imax = 501                       #: Number of real time iterations (NB: deltat = tmax/(imax-1))
sys.acon = 1.0                       #: Smoothing of the Coloumb interaction
sys.interaction_strength = 1.0       #: Scales the strength of the Coulomb interaction
sys.im = 0                           #: Use imaginary potentials (0: no, 1: yes)


def v_ext(x):
    """Initial external potential
    """
    return 0.5*(0.25**2)*(x**2)
sys.v_ext = v_ext

def v_pert(x):
    """Time-dependent perturbation potential

    Switched on at t=0.
    """
    y = -0.5*x
    if(sys.im == 1):
        return y + v_pert_im(x)
    return y
sys.v_pert = v_pert

def v_pert_im(x):
    """Imaginary perturbation potential

    Switched on at t=0.
    """
    strength = 1.0
    length_from_edge = 5.0
    I = sys.xmax - length_from_edge
    if(-sys.xmax < x and x < -I) or (sys.xmax > x and x > I):
        return -strength*1.0j
    return 0.0
sys.v_pert_im = v_pert_im


### Exact parameters
ext = InputSection()
ext.itol = 1e-12                     #: Tolerance of imaginary time propagation (Recommended: 1e-12)
ext.itol_solver = 1e-12              #: Tolerance of linear solver in imaginary time propagation (Recommended: 1e-14)
ext.rtol_solver = 1e-12              #: Tolerance of linear solver in real time propagation (Recommended: 1e-12)
ext.itmax = 2000.0                   #: Total imaginary time
ext.iimax = 1e5                      #: Imaginary time iterations
ext.ideltat = ext.itmax/ext.iimax    #: Imaginary time step (DERIVED)
ext.RE = False                       #: Reverse engineer many-body density
ext.OPT = False                      #: Calculate the external potential for the exact density
ext.excited_states = 0               #: Number of excited states to calculate (0: just calculate the ground-state)
ext.elf_gs = False                   #: Calculate ELF for the ground-state of the system
ext.elf_es = False                   #: Calculate ELF for the excited-states of the system
ext.elf_td = False                   #: Calculate ELF for the time-dependent part of the system
ext.psi_gs = False                   #: Save the reduced ground-state wavefunction to file
ext.psi_es = False                   #: Save the reduced excited-state wavefunctions to file
ext.initial_psi = 'qho'              #: Initial wavefunction ('qho' by default. 'non' can be selected. 'hf', 'lda1', 'lda2', 'lda3',
                                     #  'ldaheg' or 'ext' can be selected if the orbitals/wavefunction are available. An ext
                                     #  wavefunction from another run can be used, but specify the run.name instead e.g. 'run_name'.
                                     #: WARNING: If no reliable starting guess can be provided e.g. wrong number of electrons per well,
                                     #: then choose 'qho' - this will ensure stable convergence to the true ground-state.)


### Non-interacting approximation parameters
non = InputSection()
non.rtol_solver = 1e-14              #: Tolerance of linear solver in real time propagation (Recommended: 1e-13)
non.save_eig = True                  #: Save eigenfunctions and eigenvalues of Hamiltonian
non.RE = False                       #: Reverse-engineer non-interacting density
non.OPT = False                      #: Calculate the external potential for the non-interacting density


### LDA parameters
lda = InputSection()
lda.NE = 2                           #: Number of electrons used in construction of the LDA (1, 2, 3 or 'heg')
lda.scf_type = 'pulay'               #: how to perform scf (None, 'linear', 'pulay', 'cg')
lda.mix = 0.2                        #: Mixing parameter for linear & Pulay mixing (float in [0,1])
lda.pulay_order = 20                 #: length of history for Pulay mixing (max: lda.max_iter)
lda.pulay_preconditioner = None      #: preconditioner for pulay mixing (None, 'kerker', rpa')
lda.kerker_length = 0.5              #: length over which density fluctuations are screened (Kerker only)
lda.tol = 1e-12                      #: convergence tolerance in the density
lda.etol = 1e-12                     #: convergence tolerance in the energy
lda.max_iter = 10000                 #: Maximum number of self-consistency iterations
lda.save_eig = True                  #: Save eigenfunctions and eigenvalues of Hamiltonian
lda.OPT = False                      #: Calculate the external potential for the LDA density


### MLP parameters
mlp = InputSection()
mlp.f = 'e'                          #: f mixing parameter (if f='e' the weight is optimzed with the elf)
mlp.tol = 1e-12                      #: Self-consistent convergence tollerance
mlp.mix = 0.0                        #: Self-consistent mixing parameter (default 0, only use if doesn't converge)
mlp.reference_potential = 'non'      #: Choice of reference potential for mixing with the SOA
mlp.OPT = False                      #: Calculate the external potential for the MLP density


### HF parameters
hf = InputSection()
hf.fock = 1                          #: Include Fock term (0 = Hartree approximation, 1 = Hartree-Fock approximation)
hf.con = 1e-12                       #: Tolerance
hf.nu = 0.9                          #: Mixing term
hf.save_eig = True                   #: Save eigenfunctions and eigenvalues of Hamiltonian
hf.RE = False                        #: Reverse-engineer hf density
hf.OPT = False                       #: Calculate the external potential for the HF density


### HYB parameters
hyb = InputSection()
hyb.alpha = 1.0                      #: Fraction of HF (float in [0,1]) (set to 'o' to calculate and use optimal alpha)
hyb.alphas = (0.5,1.0,6)             #: If finding optimal alpa, this defines the range (a,b,c)  a->b in c steps
hyb.homo_occupation = 1.0            #: Occupation of the HOMO level (float in [0,1])
hyb.mix = 0.5                        #: Mixing parameter for linear  mixing (float in [0,1])
hyb.tol = 1e-12                      #: convergence tolerance in the density
hyb.max_iter = 10000                 #: Maximum number of self-consistency iterations
hyb.save_eig = True                  #: Save eigenfunctions and eigenvalues of Hamiltonian
hyb.OPT = False                      #: Calculate the external potential for the LDA density
hyb.RE = False                       #: Calculate the external potential for the LDA density


### MBPT parameters
mbpt = InputSection()
mbpt.h0 = 'non'                      #: starting hamiltonian: 'non','ha','hf','lda'
mbpt.tau_max = 40.0                  #: Maximum value of imaginary time
mbpt.tau_npt = 1001                  #: Number of imaginary time points
mbpt.norb = 35                       #: Number of orbitals to use
mbpt.flavour = 'GW'                  #: 'G0W0', 'GW0', 'GW'
mbpt.den_tol = 1e-06                 #: density tolerance of self-consistent algorithm
mbpt.max_iter = 100                  #: Maximum iterations of self-consistent algorithm
mbpt.save_full = []                  #: save space-time quantities (e.g. 'G0_iw', 'S1_it')
mbpt.save_diag = []                  #: save diaginal components of space-time quantities
mbpt.w = 'dynamical'                 #: compute 'full' W or 'dynamical' W-v
mbpt.hedin_shift = True              #: perform Hedin shift
mbpt.RE = False                      #: Reverse-engineer mbpt density
mbpt.OPT = False                     #: Calculate the external potential for the MBPT density


### LAN parameters
lan = InputSection()
lan.start = 'non'                    #: Ground-state Kohn-Sham potential to be perturbed


### RE parameters
re = InputSection()
re.save_eig = True                   #: Save Kohn-Sham eigenfunctions and eigenvalues of reverse-engineered potential
re.stencil = 5                       #: Discretisation of 1st derivative (5 or 7)
re.mu = 1.0                          #: 1st convergence parameter in the ground-state reverse-engineering algorithm
re.p = 0.05                          #: 2nd convergence parameter in the ground-state reverse-engineering algorithm
re.gs_density_tolerance = 1e-9       #: Tolerance of the error in the ground-state density
re.starting_guess = 'extre'          #: Starting guess of groud-state Vks (if not available will start with Vxt)
re.nu = 1.0                          #: Convergence parameter in the time-dependent reverse-engineering algorithm
re.rtol_solver = 1e-12               #: Tolerance of linear solver in real time propagation (Recommended: 1e-12)
re.td_density_tolerance = 1e-7       #: Tolerance of the error in the time-dependent density
re.cdensity_tolerance = 1e-7         #: Tolerance of the error in the time-dependent current density
re.max_iterations = 10               #: Maximum number of iterations per time step to find the Kohn-Sham potential
re.damping = 1.0                     #: Damping factor used when filtering out noise in the Kohn-Sham vector potential
                                     #  (0: No damping)

### OPT parameters
opt = InputSection()
opt.tol = 1e-3                       #: Tolerance of the error in the density
opt.mu = 1.0                         #: 1st convergence parameter
opt.p = 0.05                         #: 2nd convergence parameter

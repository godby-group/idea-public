"""Calculates the metric between two different wavefunctions or densities of a
2 electron system with a given external potential. An example would be calculating
the wavefunction metric for the exact and Hartree Fock. The metric would give a
measure on how different these wavefunctions were. The wavefunction metric is defined
within a range of 0 to :math:`\sqrt{N}` amd the density metric is defined within
a range of 0 to :math:`2\sqrt{2}`.
"""
import parameters as pm
from . import results as rs
import numpy as np
import scipy.sparse as sps
import scipy.misc as spmisc
import pickle
from math import *
from . import EXT_cython
import sys
import copy


def make_filename(pm):
    r"""Creates the file name for both data files containing the data needed to
    create the metric, depending on what type of metric has been requested.

    parameters
    ----------
    pm : object
        Parameters object

    returns data_file_1 and data_file_2
        variables containing a string of the relevent data files needed to
        create the desired metric.
    """

    # For density data:
    if (pm.met.type == "density"):
        data_file_1 = "gs_" + pm.met.r_type_1 + "_den"
        data_file_2 = "gs_" + pm.met.r_type_2 + "_den"

    # For wavefunction data:
    elif (pm.met.type == "wavefunction"):

        # For all wavefunctions except exact
        data_file_1 = "gs_" + pm.met.r_type_1 + "_eigf"
        data_file_2 = "gs_" + pm.met.r_type_2 + "_eigf"

        # For exact wavefunctions:
        # data 1
        if (pm.met.exact_1 == True):
            data_file_1 = "gs_" + pm.met.r_type_1 + "_psi"
        # data 2
        if (pm.met.exact_2 == True):
            data_file_2 = "gs_" + pm.met.r_type_2 + "_psi"

    # Error checking:
    else:
        sys.exit("met: Warning: metric type not properly defined. Either 'density' or"
                 " 'wavefunction'. Exiting.")


    pm.sprint("met: datafiles: gs_{0}/{1}/ and gs_{2}/{3}".format(pm.met.r_type_1 \
              ,data_file_1,pm.met.r_type_2,data_file_2))
    return data_file_1,data_file_2


def construct_expansion_matrix(pm):
    r"""Constructs the expansion matrix that is used to expand the reduced 
    wavefunction (insert indistinct elements) to get back the full 
    wavefunction.

    parameters
    ----------
    pm : object
        Parameters object

    returns sparse_matrix
        Expansion matrix used to expand the reduced wavefunction (insert
        indistinct elements) to get back the full wavefunction.
    """
    # Number of elements in the reduced wavefunction
    coo_size = int(np.prod(list(range(pm.space.npt,pm.space.npt+pm.sys.NE)))\
               /spmisc.factorial(pm.sys.NE))
    
    # COOrdinate holding arrays for the expansion matrix 
    coo_1 = np.zeros((pm.space.npt**pm.sys.NE), dtype=int)
    coo_2 = np.copy(coo_1)
    coo_data = np.zeros((pm.space.npt**pm.sys.NE), dtype=np.float)  

    # Populate the COOrdinate holding arrays with the coordinates and data
    if(pm.sys.NE == 2):
        coo_1, coo_2, coo_data = EXT_cython.expansion_two(coo_1, coo_2, 
                                 coo_data, pm.space.npt)
    elif(pm.sys.NE == 3):
        coo_1, coo_2, coo_data = EXT_cython.expansion_three(coo_1, coo_2, 
                                 coo_data, pm.space.npt)

    # Convert the holding arrays into COOrdinate sparse matrices
    expansion_matrix = sps.coo_matrix((coo_data,(coo_1,coo_2)), shape=(
                       pm.space.npt**pm.sys.NE,coo_size), dtype=np.float)

    # Convert into compressed sparse row (csr) form for efficient arithemtic
    expansion_matrix = sps.csr_matrix(expansion_matrix)

    return expansion_matrix


def load_data(pm):
    r"""Loads data_file_1 and data_file_2 into a numpy array using pickle.load().
       It does this by opening the full filepath using the run name set by the
       user as a parameter.

    parameters
    ----------
    pm : object
        Parameters object

    returns raw_data_1 and raw_data_2
        numpy arrays containing the raw data required to create the metrics.
    """

    # Function call to generate the filename:
    data_file_1, data_file_2 = make_filename(pm)

    # Using pickle to load in the raw data for both runs:
    with open("outputs/" + str(pm.met.r_name_1) + "/raw/" + str(data_file_1) \
              + ".db",'rb') as metric_file_1:
        raw_data_1 = pickle.load(metric_file_1)

    with open("outputs/" + str(pm.met.r_name_2) + "/raw/" + str(data_file_2) \
              + ".db",'rb' ) as metric_file_2:
        raw_data_2 = pickle.load(metric_file_2)

    return raw_data_1,raw_data_2


def slat_calc(pm,raw_data):
    r"""Calculates the slater determinant of the eigenfunctions (single particle
    orbitals), which is the total wavefunction. It is the way of calculating a
    wavefunctions for non-exact methods like HF and LDA, which is required for
    calculating a wavefunction metric with data from these approximations. This
    function is only necessary if it is the wavefunction metric that is being
    calculated and is non exact, as the exact wavefunction is calculated in
    EXT2.py.

    parameters
    ----------
    pm : object
        Parameters object
    raw_data : numpy array
        the eigenfunctions of all states of the system. Only first two occupied
        states are needed (raw_data[0] and raw_data[1]).

    returns psi
        numpy array containing the Slater determinant wavefunction.
    """

    # Gathering appropriate eigenfunctions (first two occupied states):
    phi_1 = raw_data[0]
    phi_2 = raw_data[1]
    # Zeroing determinant array:
    det   = np.zeros((pm.sys.grid,pm.sys.grid),dtype=complex)

    #Calculating the determinant:
    for i in range(pm.sys.grid):
        for j in range(pm.sys.grid):
            det[i,j] = phi_1[i]*phi_2[j] - phi_1[j]*phi_2[i]

# Normalising the wavefunctions - sqrt(2) factor from Irene's normalisation
# convention and 1/sqrt(2) from the standard normalisation convention cancel
# out for slater determinant wavefunctions.
    psi = det
    return psi


def data_prep(pm,raw_data_1,raw_data_2):
    r"""Prepares the data ready for the metric calculation, which is different
    depending on the type of metric and the type of data. If either of the raw
    data contain the exact wavefunction, it will be renormalised using a different
    convention that is required for metrics: instead of being normalised to 1,
    the wavefunction is normalised to :math:`N`. If the raw data is a non-exact
    wavefunction, slat_calc() is called. If the raw data is the density, it is
    already in the required form so this function does nothing to it.

    parameters
    ----------
    pm : object
        Parameters object
    raw_data_1 : numpy array
        Contains either the exact wavefunction, the single_particle
        eigenfunctions if its a non-exact method, or the density of the 1st
        system.
    raw_data_2 : numpy array
        Contains either the exact wavefunction, the single_particle
        eigenfunctions if its a non-exact method, or the density of the 2nd
        system.

    returns data_1 and data_2
        The data required to calculate the metric, now ready to be plugged into
        either the density or the wavefunction metric equation.
    """

    # Setting up wavefunction data:
    if (pm.met.type == "wavefunction"):

        # Checking whether data is exact:
        # 1st data - exact
        if (pm.met.exact_1 == True):
            expansion_matrix = construct_expansion_matrix(pm)
            data_1 = sqrt(2)*expansion_matrix*raw_data_1
            pm.sprint("met: 1st system: exact")
        # 1st data - not exact
        elif (pm.met.exact_1 == False):
            data_1 = slat_calc(pm,raw_data_1)
            pm.sprint("met: 1st system: not exact")
        # 2nd data - exact
        if (pm.met.exact_2 == True):
            expansion_matrix = construct_expansion_matrix(pm)
            data_2 = sqrt(2)*expansion_matrix*raw_data_2
            pm.sprint("met: 2nd system: exact")
        # 2nd data - not exact
        elif (pm.met.exact_2 == False):
            data_2 = slat_calc(pm,raw_data_2)
            pm.sprint("met: 2nd system: not exact")

    # Setting up density data:
    elif (pm.met.type == "density"):
        data_1 = raw_data_1
        data_2 = raw_data_2

    return data_1,data_2


def mat_calc(pm,data_1,data_2):
    r"""Calculates either the density or wavefunction metrics, depending on the
    input data, using the defined metric equations.

    .. math::

            \text{wavefunction metric}: \ &D_{\psi}(\psi_1,\psi_2) = \sqrt{\int}(|\psi_1|^2+|\psi_2|^2)dx -2|\int\psi_{1}^{*}\psi_2 dx|\\
            \text{density metric}: \ &D_{\ro}(\ro_1,\ro_2) = \int |\ro_1 -\ro_2|^2 \\ \\

    parameters
    ----------
    pm : object
        Parameters object
    data_1 : numpy array
        Contains either the normalised exact wavefunction, the slater deterimant
        wavefunction, or the density of the 1st system.
    data_2 : numpy array
        Contains either the normalised exact wavefunction, the slater deterimant
        wavefunction, or the density of the 2nd system.

    returns metric
        The metric value for the two sets of data. This is a scalar between then
        set bounds.
    """

    metric = 0
    grid   = pm.sys.grid
    dx     = 2.0*pm.sys.xmax/(pm.sys.grid-1)
    # Employing the wavefunction metric equation
    if (pm.met.type == "wavefunction"):
        for k in range(grid):
            for l in range(grid):
                metric += (abs(data_1[k,l])**2 + abs(data_2[k,l])**2)*dx \
                            - 2*abs(np.conjugate(data_1[k,l])*data_2[k,l])*dx

        metric = np.sqrt(metric)

    # Employing the density metric equation
    if (pm.met.type == "density"):
        metric = 0.5*sum(abs(data_1-data_2))*dx

    return metric


def main(pm):
    r"""Calculates the desired metric, with the input files being set as a
    parameter. Calls the necessary functions defined above to do this.

    parameters
    ----------
    parameters : object
        Parameters object

    returns object
        Results object
    """
    #Check that number of electrons = 2 (only works for 2)
    if (pm.sys.NE != 2):
        pm.sprint("met: Warning: number of electrons not 2 (only works for 2)")

    # Contains all the function calls.
    pm.sprint("met: metric type: {}".format(pm.met.type))

    # STEP 1 - load in raw data from iDEA:
    raw_data_1, raw_data_2 = load_data(pm)
    # STEP 2 - prepare data for metric calculation
    data_1, data_2 = data_prep(pm,raw_data_1,raw_data_2)
    # Step 3 - calculate metric:
    pm.sprint("met: calculating metric...")
    metric = mat_calc(pm,data_1,data_2)
    pm.sprint("met: metric value:{}".format(metric))

    # Check for wavefunction metric being within legal bounds:
    if (pm.met.type == "wavefunction"):
        if (metric < 0.0 or metric > 2*sqrt(2.0)):
            pm.sprint("met: Warning: metric value outside bounds; non physical")

    # Check for density metric being within legal bounds
    if (pm.met.type == "density"):
        if (metric < 0.0 or metric > 4.0):
            pm.sprint("met: Warning: metric value outside bounds; non physical")
    # Add to results object
    results = rs.Results()
    if(pm.met.type == 'density'):
        t = 'den'
    if(pm.met.type == 'wavefunction'):
        t = 'wav'
    results.add(metric, 'gs_{0}_{1}_{2}_{3}_{4}met'.format(pm.met.r_name_1, \
                pm.met.r_type_1, pm.met.r_name_2, pm.met.r_type_2, t))

    if pm.run.save:
       results.save(pm)

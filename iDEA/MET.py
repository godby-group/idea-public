#==============================Metric code for iDEA============================#
#-----------------------------Ewan Richardson-22/10/17-------------------------#
# A code which generates ground state density and wavefunction metrics using the
# iDEA code. Developed as part of my MPhys project.
#-----------------------------------Libraries----------------------------------#
import parameters as pm
from . import results as rs
import numpy as np
import pickle
from math import *
import sys
import copy
#----------------------------------FUNCTIONS-----------------------------------#
#--------------------------Loading in Data from iDEA---------------------------#
def make_filename(pm):
# Creates the filenames of the data used to generate metrics.

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
            data_file_1 = "gs_" + pm.met.r_type_1 + "_psi_exp"
        # data 2
        if (pm.met.exact_2 == True):
            data_file_2 = "gs_" + pm.met.r_type_2 + "_psi_exp"

# Error checking:
    else:
        sys.exit("met: Warning: metric type not properly defined. Either 'density' or"
                 " 'wavefunction'. Exiting.")


    pm.sprint("met: datafiles: gs_{0}/{1}/ and gs_{2}/{3}".format(pm.met.r_type_1 \
              ,data_file_1,pm.met.r_type_2,data_file_2))
    return data_file_1,data_file_2

def load_data(pm):
# Loads data from selected files into a numpy array.

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
#----------------------Calculating the Slater Determinant----------------------#
def slat_calc(pm,raw_data):
# Calculate the wavefunctions, that are not exact, from a Slater determinant.

# Gathering appropriate eigenfunctions (first two occupied states):
    phi_1 = raw_data[0]
    phi_2 = raw_data[1]
# Zeroing determinant array:
    det   = np.zeros((pm.sys.grid,pm.sys.grid))

# Calculating the determinant:
    for i in range(pm.sys.grid):
        for j in range(pm.sys.grid):
            det[i,j] = phi_1[i]*phi_2[j] - phi_1[j]*phi_2[i]

# Normalising the wavefunctions - sqrt(2) factor from Irene's normalisation
# convention and 1/sqrt(2) from the standard normalisation convention cancel
# out for slater determinant wavefunctions.
    psi = det
    return psi
#------------------------------Preparing the Data------------------------------#
def data_prep(pm,raw_data_1,raw_data_2):
# This function does multiple things, depending on the type of data.
# Wavefunction -     exact: normalising using Irene's convention
# Wavefunction - non exact: calling Slater determinant calculation function
#      Density -          : raw data is in correct form so no prep necessary

# Setting up wavefunction data:
    if (pm.met.type == "wavefunction"):

# Checking whether data is exact:
        # 1st data - exact
        if (pm.met.exact_1 == True):
            data_1 = sqrt(2)*raw_data_1
            pm.sprint("met: 1st system: exact")
        # 1st data - not exact
        elif (pm.met.exact_1 == False):
            data_1 = slat_calc(pm,raw_data_1)
            pm.sprint("met: 1st system: not exact")
        # 2nd data - exact
        if (pm.met.exact_2 == True):
            data_2 = sqrt(2)*raw_data_2
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
#----------------------------Calculating the Metrics---------------------------#
def mat_calc(pm,data_1,data_2):

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
#---------------------------------MAIN FUNCTION--------------------------------#
def main(pm):
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

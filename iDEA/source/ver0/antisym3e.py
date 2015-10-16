######################################################################################
# Name: Antisymmetrisation operations for 3 electrons matrices                       #
######################################################################################
# Author(s): James Ramsden                                                           #
######################################################################################
# Description:                                                                       #
# Reduces the size of the many body matrices for 3 electron systems by using the     #
# Pauli exclusion principle.                                                         #
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

# Library imports
from numpy import prod, zeros
from scipy.misc import factorial
from scipy.sparse import lil_matrix

# Construct grid
def gridz(N_x):
    N_e = 3
    Nxl = N_x**N_e
    Nxs = prod(range(N_x,N_x+N_e))/int(factorial(N_e))
    sgrid = zeros((N_x,N_x,N_x), dtype='int')
    count = 0
    for ix in range(N_x):
        for jx in range(ix+1):
            for kx in range(jx+1):
                sgrid[ix,jx,kx] = count
                count += 1
    lgrid = zeros((N_x,N_x,N_x), dtype='int')
    count = 0
    for ix in range(N_x):
        for jx in range(N_x):
            for kx in range(N_x):
                lgrid[ix,jx,kx] = count
                count += 1
    return sgrid, lgrid

# Construct antisym matrices
def antisym(N_x):
    N_e = 3
    Nxl = N_x**N_e
    Nxs = prod(range(N_x,N_x+N_e))/int(factorial(N_e))
    sgrid, lgrid = gridz(N_x)
    C_down = lil_matrix((Nxs,Nxl))
    for ix in range(N_x):
        for jx in range(ix+1):
            for kx in range(jx+1):
                C_down[sgrid[ix,jx,kx],lgrid[ix,jx,kx]] = 1.
    C_up = lil_matrix((Nxl,Nxs))
    for ix in range(N_x):
        for jx in range(ix+1):
            for kx in range(jx+1):
                C_up[lgrid[ix,jx,kx],sgrid[ix,jx,kx]] = 1.
                C_up[lgrid[ix,kx,jx],sgrid[ix,jx,kx]] = -1.
                C_up[lgrid[jx,ix,kx],sgrid[ix,jx,kx]] = -1.
                C_up[lgrid[kx,jx,ix],sgrid[ix,jx,kx]] = -1.
                C_up[lgrid[kx,ix,jx],sgrid[ix,jx,kx]] = 1.
                C_up[lgrid[jx,kx,ix],sgrid[ix,jx,kx]] = 1.
                C_up[lgrid[jx,kx,ix],sgrid[ix,jx,kx]] = 1.
                C_up[lgrid[kx,ix,jx],sgrid[ix,jx,kx]] = 1.
                C_up[lgrid[jx,kx,ix],sgrid[ix,jx,kx]] = 1.
                C_up[lgrid[kx,ix,jx],sgrid[ix,jx,kx]] = 1.
    C_down = C_down.tocsr()
    C_up = C_up.tocsr()
    return C_down, C_up


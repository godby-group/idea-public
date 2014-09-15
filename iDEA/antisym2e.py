from numpy import prod, zeros, delete, random, linspace, dot, ones, array, sqrt
from scipy.misc import factorial
from scipy.sparse import lil_matrix, identity, spdiags

def gridz(N_x):
    N_e = 2
    Nxl = N_x**N_e
    Nxs = prod(range(N_x,N_x+N_e))/int(factorial(N_e))
    sgrid = zeros((N_x,N_x), dtype='int')
    count = 0
    for ix in range(N_x):
        for jx in range(ix+1):
            sgrid[ix,jx] = count
            count += 1
    lgrid = zeros((N_x,N_x), dtype='int')
    lky = zeros((Nxl,2), dtype='int')
    count = 0
    for ix in range(N_x):
        for jx in range(N_x):
            lgrid[ix,jx] = count
            lky[count,0] = ix
            lky[count,1] = jx
            count += 1
    return sgrid, lgrid

def antisym(N_x, retsize=False):
    N_e = 2
    Nxl = N_x**N_e
    Nxs = prod(range(N_x,N_x+N_e))/int(factorial(N_e))
    sgrid, lgrid = gridz(N_x)
    C_down = lil_matrix((Nxs,Nxl))
    for ix in range(N_x):
        for jx in range(ix+1):
            C_down[sgrid[ix,jx],lgrid[ix,jx]] = 1.
    C_up = lil_matrix((Nxl,Nxs))
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

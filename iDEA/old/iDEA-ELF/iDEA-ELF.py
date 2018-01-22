"""    print
    print'                  *    ****     *****       *                  '
    print'                       *   *    *          * *                 '
    print'                  *    *    *   *         *   *                '
    print'                  *    *     *  *****    *     *               '
    print'                  *    *    *   *       *********              '
    print'                  *    *   *    *      *         *             '
    print'                  *    ****     ***** *           *            '
    print'      ______ _      ______   _    _      _                     '
    print'     |  ____| |    |  ____| | |  | |    | |                    '
    print'     | |__  | |    | |__    | |__| | ___| |_ __   ___ _ __     '
    print'     |  __| | |    |  __|   |  __  |/ _ \ | \'_ \ / _ \ \'__|    '
    print'     | |____| |____| |      | |  | |  __/ | |_) |  __/ |       '
    print'     |______|______|_|      |_|  |_|\___|_| .__/ \___|_|       '
    print'                                          | |                  '
    print'                                          |_|                  '
    print
    print '  ------------------------------------------------------------'
    print '  |                        iDEA - ELF                        |'
    print '  |          (Interacting Dynamic Electrons Approach         |'
    print '  |              to Electron Localisation Functions)         |'
    print '  |                   Created by Tom Durrant                 |'                           
    print '  |                   The University of York                 |'
    print '  ------------------------------------------------------------'
    print
    print 'Requirements: Charge density of the interacting system.      '
    print '              Kohn-Sham Orbitals from iDEA-TDDFT             '
    print '              Parameters file used by the interacting system.' """

# At the moment, only getElf has been generalised to three electron systems.
# All other functions work for two electrons.
#if  __name__ == '__main__': 


import numpy as np
from matplotlib import pyplot as plt
import os
import imp
import scipy.optimize as opt

locDir = os.getcwd()
print "ELF load hunting for parameters module in"
print locDir
try:
    pm = imp.load_module("pm", *imp.find_module("parameters3", [locDir]))
except ImportError:
    try:
        pm = imp.load_module("pm", *imp.find_module("parameters", [locDir]))
    except:
        print "Error: Could not find parameters module"
        #raise
def getElf(den, KS, j=None, posDef=False):
    """ Calculate ELF from the Savin et al formulation.

    den is the MB charge density calculated by iDEA-MB.

    KS is a list of the Kohn-Sham wavefunctions.
    ( ie KS = [ks1, ks2, ks3] )
    where ks1 ect are numpy arrays.

    If the current density j is provided then the 
    extra time-dependent component is included.
    
    posDef is an optional argument that forces a positive
    definite approximation to c and is not a part of the 
    usual formulation.    """

    # The single particle kinetic energy density terms
    grad1 = np.gradient(KS[0], pm.deltax)
    grad2 = np.gradient(KS[1], pm.deltax)
    if len(KS) > 2:
        grad3 = np.gradient(KS[2], pm.deltax)
    else:
        grad3 = 0.0

    # Gradient of the density
    gradDen = np.gradient(den, pm.deltax)

    # Unscaled measure
    c = np.arange(den.shape[0])
    if j == None:
        c = (np.abs(grad1)**2 + np.abs(grad2)**2 + np.abs(grad3)**2)  \
            - (1./4.)* ((np.abs(gradDen)**2)/den)
    elif (j.shape == den.shape):
        c = (np.abs(grad1)**2 + np.abs(grad2)**2 + np.abs(grad3)**2)  \
            - (1./4.)* ((np.abs(gradDen)**2)/den) - (j**2)/(den)
  
    else:
        print "Error: Invalid Current Density given to ELF"
        print "       Either j wasn't a numpy array or it "
        print "       was of the wrong dimensions         "
        return None
    
    # Force a positive-definate approximation if requested
    if posDef == True:
        for i in range(den.shape[0]):
            if c[i] < 0.0:
                c[i] = 0.0
    
    elf = np.arange(den.shape[0])

    # Scaling reference to the homogenous electron gas
    c_h = getc_h(den)

    # Finaly scale c to make ELF
    elf = (1 + (c/c_h)**2)**(-1)

    return elf

    
    
def cToElf(den, c):
    """ Calculate ELF given c"""

    elf = np.arange(den.shape[0])

    # Scaling reference to the homogenous electron gas
    c_h = getc_h(den)

    elf = (1 + (c/c_h)**2)**(-1)

    return elf    
 
 
 
def getc_h(den):
    """ C for the 1D electron gas. Used as a scaling reference"""
    c_h = np.arange(den.shape[0])

    c_h = (1./6.)*(np.pi**2)*(den**3)

    return c_h
    
    

def sphericalPair(psi):
    """ Take a wavefunction and convert it into a radialy averaged Conditional Pair Density.
    the returned function has the form P(x,s) where x is the postion of a reference
    electron and s is a radius. spherPair[x,:] is normalised to 1 for all x"""
    
    den = getDen(psi)
    pairDen = getPairDen(psi)
    spherPair = np.zeros(pairDen.shape)
    #Radially average
    for x1 in range(pairDen.shape[0]):
        spherPair[x1, 0] = pairDen[x1,x1]
        for s in range(1,pairDen.shape[0]):
            s1 = x1 + s
            s2 = x1 - s
            #Division by den[x1] converts to conditional prob
            if (0<=s1<pairDen.shape[0]):
                spherPair[x1, s] += 0.5*pairDen[x1,s1]/den[x1]
            if (0<=s2<pairDen.shape[0]):
                spherPair[x1, s] += 0.5*pairDen[x1,s2]/den[x1]
        
    return spherPair
    

    
def quadratic(x, A):
    """ A quadratic of the form A*x**2 """
    return (x**2)* A 
    
    
    
def exactElf(psi):
    """ Calculate my "exact" elf from a wavefunction """
  
    # Construct modPair, the function to fit
    sPair = sphericalPair(psi)
    modPair = (sPair)
    modPair[:,0] = 0.0
   
    cExact = c_Exact(psi)
            
    # Convert this exact c into an ELF    
    exact_elf  = cToElf(den, cExact)
    
    return exact_elf
    
def c_Exact(psi):
    """ Calculate exact ELF using my method with an exact fit to 
    the spherical pair den. Use the Dobson method instead!"""
    sPair = sphericalPair(psi)
    modPair = (sPair)
    modPair[:,0] = 0.0
    cExact = np.zeros(modPair.shape[0])

    # For every value of x fit a polynomial and get "exact" c
    for x in range(sPair.shape[0]-1):
        fitNo = 3 # Number of datapoints to use
        
        xSection = np.linspace(0, fitNo-1, fitNo)*pm.deltax
        xAll = np.linspace(0, pm.jmax, pm.jmax)*pm.deltax
        fitSection = modPair[x, 0:fitNo]
        coefs, errors = opt.curve_fit(quadratic, xSection, fitSection)

        cExact[x] = coefs
        
        cERange = np.zeros(pm.jmax)
        for s in range(pm.jmax):
            sR = s*pm.deltax
            cERange[s] = coefs*(sR**2)
        

        if False: # Plot interactive fit
            plt.ion()
            plt.cla()
            plt.plot(xSection, fitSection, label="Exact fit region")
            plt.plot(xAll, cERange, label="Fit")
            plt.plot(xAll, modPair[x, :], label="Exact")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=9, ncol=2, mode="expand", borderaxespad=0.)
            plt.xlabel("s")
            plt.ylabel("Radially Averaged Conditional Probability")
            plt.title("x={}".format(x*pm.deltax))
            plt.draw()
            plt.pause(0.1)
      
    return cExact
    

def getDen(psi):
    """ Get the density of a two or three electon system """
    den1 = np.zeros(psi.shape[0])
    
    # Work out how many electrons we are working with
    N = psi.ndim
    print "Debug: N is", N
    
    # Integrate out x3 (if this co-ordinate exists)
    if N > 2:
        den2 = np.zeros([psi.shape[0], psi.shape[0]])
        for i in range(0, psi.shape[0]):
            for j in range(0, psi.shape[0]):
                den2[i, j] = np.sum(np.abs(psi[i,j,:])**2) * (pm.deltax)
    else:
        den2 = np.abs(psi**2)
    
    # Integrate out x2
    for i in range(0, psi.shape[0]):
        den1[i] = np.sum(den2[i,:]) * (pm.deltax)
        
    # Normalise to the particle number
    den1 = den1 * N
    
    print "Charge Density normalised to ", np.sum(den1) * pm.deltax, " (Should be {})".format(N)
    
    return den1
    
def getPairDen(psi):
    """ Get the pair density of a two or three electron system  """
    
    N = psi.ndim

    # Two electron system
    if psi.ndim == 2:
        pairDen = N*(N-1)*np.abs(psi**2)
    
    # Three electron system
    elif psi.ndim == 3:
        pairDen = np.zeros([pm.jmax, pm.jmax])
        triDen = np.abs(psi)**2
        
        for i in range(0, psi.shape[0]):
            for j in range(0, psi.shape[0]):
                pairDen[i, j] = N*(N-1)*np.sum(triDen[i,j,:]) * (pm.deltax)
    
    integral = np.sum(pairDen[:,:])*(pm.deltax**2)
    
    print "Debug: Pair Den consitency check Integral=", integral, " (Should be {})".format(N*(N-1))
                
    #print "Pair Density normalised to ", np.sum(pairDen) * pm.deltax**2, " (Should be 1)"
    
    
    
    return pairDen
    
def getCDobson(psi):
    """ Use the atternative form of C from Dobson - Interpritation of the Fermi Hole Curveture
    Alternative calculation from wavefucntion. Scaled differently to other C 
    C_Dob = c/2n """
    
    pairDen = getPairDen(psi)
    # PairDen(r, rPrime)
    laplacianPrime = np.gradient(np.array(np.gradient(pairDen, pm.deltax)[1]), pm.deltax)[1]
    
    c = np.zeros(pm.jmax)
    
    for r in range(0, pm.jmax):
        c[r] = laplacianPrime[r, r]

    return c
    
def getC_hDobson(den):
    """ Get the scaling reference from the homogenous electron gas using
    the Dobson defintion """
    c_h = np.arange(den.shape[0])

    c_h = (1./3.)*(np.pi**2)*(den**4)

    
    return c_h

def getElfDobson(psi, square=True):
    """ Calculate the exact ELF using the Dobson formulation
    argument is the wavefunction """
    
    den = getDen(psi)
    
    c = getCDobson(psi)
    c_sigma = c/(2.0*den)
    
    
    c_fit = c_Exact(psi)
    cost = np.sum(np.abs(c_sigma-c_fit))*pm.deltax

    print "Debug: Fit vs Dobson Cost is {} (Should be 0)".format(cost)
    print "Debug: Dobson weight is", np.sum(np.abs(c_sigma))
    print "Debug: Fitted weight is", np.sum(np.abs(c_fit))
    
    c_h = getC_hDobson(den)
    
    elf = np.zeros(psi.shape[0])
    
    if square == True:
        elf = 1.0 / (1.0 + (c/c_h)**2)
    elif square == False:
        elf = 1.0 / (1.0 + abs(c/c_h))
    else:
        raise TypeError("getElfDobson: square is not a Boolean")
    
    return elf
    
def elfIntegral(den, elf):
    """ My Attempt at turning elf into a system wide 
    number like my localisation measure 
    
    N is the particle number"""
    
    integrand = elf * den
    
    elfLoc = np.trapz(integrand, dx=pm.deltax)
    
    N = (np.sum(den)*pm.deltax)
    
    return elfLoc/N
    
    
def RELP(psi):
    """Caculate the percentage chance that each region contains
    one electron. Uses findBoundaries to place the region seperations """
    relp = None
    return relp
    

def RELM(psi):
    """Caculate RELP and convert to a form for direct comparison with elf integrals """
    relm = None
    return relm
    
def findBoundaries(psi):
    """"Find localisation regions by intergrating the charge density into regions
    containg one. Moves from left to right and places the dividers at the integer
    electron points. """
    return a, b

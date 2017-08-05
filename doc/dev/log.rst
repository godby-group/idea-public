Changelog
=========
 * **v2.2.0** (2017-07-07)

   * HF: fixed convention used in Fock operator & 10x speedup from faster
     construction. fock() now returns F, not F*dx.
   * HF: fixed bug in mixing
   * HF: added time-dependence
   * LDA: major rewrite. Added Pulay mixing and conjugate gradient methods
     for converging tricky systems (Pulay is now the default)
   * EXT2/3: construction of initial wave function is faster
     (~2x speed up in EXT2, ~6x in EXT3)
   * EXT2/3: Optimisation code rewritten, new "OPT" section in the
     input parameters
   * EXT2/3: 2-3x speedup in imaginary time via better starting wave function.
     Another 3x by improved construction of arrays and matrices.
     Gram-Schmidt algorithm implemented to allow calculation of excited states.
     Option to save wavefunctions.
   * EXT2/3: Initial wave function can now be constructed from HF, LDA or NON 
     orbitals (harmonic oscillator remains the default).
     Alternatively, an EXT wave function can be given from a previous run.
   * RE: Now looks for an exact KS potential to start from. 
   * MBPT: Can now save 
     the "many-body self-energy" iGW + V_h0 - V_h (+ Hedin shift) (S),
     the "GW self-energy" iGW (Sxc) as well as the exchange-only
     (Sx) and correlation-only parts (Sc).
   * Added "space grid" class.
     The Input object now provides some useful grids that should be
     useful in almost any iDEA calculation

     pm.space.grid    # the spatial grid
     pm.space.v_ext   # the external potential on the grid
     pm.space.v_int   # the coulomb interaction on the grid
   * LDA: Fixed inconsistency in LDA E_xc
     Mike originally fitted the e_xc(n) and then put in
     (numerically computed) values for V_xc(n) that were
     capped after the 2nd or 3rd digit.

     We now use only *one* set of parameters, with all others
     derived directly from them (accurate to machine precision).
   * iDEA.py renamed to run.py
     The name iDEA.py results in conflicts when resolving import statements
     (in particular, when copying iDEA.py to another directory, you
     received an error when running `python iDEA.py`).
   * Makefile: Now contains a "make clean" command that should work cross-platform
   * Pickled version of parameters object is now saved in outputs/run_name/parameters.p
   * Added "stencil" parameter that specifies how many space points are taken
     to discretise the 2nd derivative (for the kinetic energy).
     Added to most codes (EXT, LDA, LAN, MLP, RE, HF, NON)
   * Results object now has .save_hdf5 function which can store all results
     in a single HDF5 file (HDF5 is a widely supported, descriptive format
     that can be read on many OS's and programming languages)
   * EXT1: drop-in replacement for SPiDEA for 1e runs.
     Solves 1e Schroedinger equation via eigensolver (no imaginary time).
     Excited states can be calculated. The ground state can be perturbed and 
     propagated through real time via Crank-Nicholson.
   * ELF: option to take density, if known
   * MKL dependency has been removed completely.
     EXT2 was found to be slower with MKL than with numpy/scipy routines in Anaconda.
   * MBPT: 400x speedup in extrapolation of the Green function
   * All of iDEA is now compatible with python3.
     python3 is now the default python version used in York in view of
     discontinuation of official python2 support in 2020. As of now, iDEA should
     still be backwards compatible with python2 though.
     python version needs to be specified in the "architecture" file in arch/
   * automatic test of unit test coverage via "coverage" python module
     (see documentation)
   * ViDEO: now uses matplotlib for plotting   
     (except for animations, which haven't been modified)
   * orbitals for LDA, NON and HF are now save by default
   * Order of runs has changed: EXT is run after the various approximations,
     thus making it more convenient to select one of them as a starting point.
   * Current density is now always zero at grid point 0
     (shifted current density by one grid point to the right).
     Fixed bug for imaginary potentials.
   * All pickle calls use protocol 4 (not default in python3)

 * **v2.1.0** (2017-02-15)

   * specify maximum number of iterations for LDA
     (Warning issued, if self-consistency is not achieved)
   * MBPT: now properly handles complex starting orbitals
   * HF: output orbitals correctly normalized
   * LDA: added "direct energy method"
   * SPiDEA: fixed deltat not being computed correctly,
     xgrid was not properly defined for starting orbitals
   * Added ELF code (ground state + time dependence)
     for exact calculations
   * EXT2: tidied up, removed global variables
   * added scripts/plot_3d.py to plot space-time quantities (MBPT)
   * EXT2: fixed bug in time-dependence, now uses arrays instead of
     lists
   * When importing iDEA, Fortran libraries are recompiled automatically, 
     if out of date
   * MBPT: Hedin shift now aware of orbital (re-)ordering
   * EXT2 & EXT3: Reduction and expansion matrices now constructed in Fortran
     (~150x speedup). Initial wave function for imaginary time propagation
     now constructed in Fortran (~100x speedup). Python functions rewritten
     (~5x speedup for imaginary time propagation).
     Some speedup in real time, but still main bottleneck of the code
   * For input section parameters that are not specified in the parameters file,
     iDEA will now take the default values, as specified in input.py
   * RE: fixed bug for time-dependent RE
   * HF/LDA: removed global variables

 * **v2.0.0** (2017-01-06)

   * fixed reading of total energy in RE
   * NON, HF and LDA can now save eigenvalues
   * added unit test for EXT2
   * re-implemented MBPT, major speedup, now using unshifted time-grid
     (using extrapolation to compute G(0+))
   * eliminated "job" class
   * added unit test for extrapolation of G
   * added GW0 approximation
   * RE,LDA,HF,NON now save orbitals (eigf) and energies (eigv)
   * Added docstrings to various codes
   * moved ViDEO to scripts subdirectory

 * **v2.0b** (2016-11-08)

   * Enables multiple runs of iDEA from one python script (e.g. to
     perform convergence tests). The parameters file is no longer imported by
     every part of iDEA, it is imported once and then passed on. 
     
   * iDEA is now structured like a regular python package.
  
   * You can now process results from iDEA calculations directly in a python
     script. All codes return a "results" object that contains computed
     quantities (e.g. results.NON.gs_non_den for the ground state density of
     the non-interacting system)
  
   * Documentation web site http://www.cmt.york.ac.uk/group_info/group/ideav2/
     uses sphinx to generate the website directly from the simple "restructured
     text" format. The source .rst files used to generate the web page are part
     of the git repository, everybody can contribute sections to the web page
     by directly editing the .rst files.
     The documentation web site includes an API documentation that is
     automatically generated from the iDEA code, where the code follows python
     standards.
  
   * Simple unit test for NON added. In order to add unit tests for your parts
     of the code, simply follow this example (or ask for advice).
  
   * While iDEA can be run in the same way as before (preparing a parameters.py
     file and running "python iDEA.py"), there are now other possibilities.
     "examples" directory has ex01, ex02, ex03 demonstrating different ways of
     running iDEA.


 * **v1.9.0** (2016-09-07)

   - EXT2 matrix construction optimised using f2py
   - MBPT code now uses offset grid
   - MLP is now time-dependent 

 * **v1.8.1** (2016-08-09)

   - MBPT code bug fixed (now works correctly with different starting orbitals)
   - Time dependent MLP added
   - Energy bug in EXT2 and EXT3 fixed

 * **v1.8.0** (2016-07-29)

   - Mike's LDA codes replaced with Matt's (~1000x speedup)
   - Danielle's MLP codes replaced with Matt's (optimised, works with external as reference)
   - Fixed bug in MB3 (now outputs density)
   - Renamed Many Body (MB) to exact (EXT) (so codes are now iDEA_EXT2,iDEA_EXT3)
   - MBPT code bug fixed (now works correctly with different starting orbitals)
   - Tested some less used parts of iDEA (and they work!)

 * **v1.7.0** (2016-07-15)

   - Landauer code added
   - Hartree approximation added (parameter added to HF code parameters)
   - MB2 and MB3 optimised
   - LDA code cleaned up
   - ViDEO now outputs HD videos and plots (1920x1080)
   - GW code now supports different starting orbitals
   - GW code parameters now converged (only in HF regime be sure to check!)

 * **v1.5.1** (2016-06-16)

   - MLP updated

 * **v1.5.0** (2016-06-16)

   - Added Many-Body Perturbation Theory code (GW approximation). MBPT can now
     be run to generate densities and Kohn-Sham potentials and is compatible
     with ViDEO and Reverse Engineering in the same way as the rest of the
     codes.
   - Added Hartree-Fock code. HF is now integrated into iDEA and is compatible
     with ViDEO and Reverse Engineering in the same way as the rest of the
     codes.
   - Other small clean-ups

 * **v1.4.1** (2016-04-05)

   - Fixed major bug in MB2 and MB3 introduced in version 1.4.0 causing the code to crash when attempting to output the external potential.

 * **v1.4.0** (2016-04-03)

   - Imaginary potentials have been added to all parts of iDEA and tested.

 * **v1.3.3** (2016-03-11)

   - iDEA-RE now allows the user to quit before the time-dependent simulation is complete, whilst still outputting the potential/density etc.

 * **v1.3.2** (2016-03-08)

   - Made the calculation of the current density much more efficient.
   - Add the calculation of the current density to 3-electron many body, 
     non-interacting approximation and the LDA.

 * **v1.3.1** (2016-02-28)

   - LDA made usable for any number of electrons
   - Add the calculation of the current density to Many-Body (no need to run reverse-engineering)
   - Fixed some minor bugs in Many-Body 3 (Time dependence)
   - General Cleanup

 * **v1.3.0** (2016-02-15)

   - Reverse engineering time dependence fixed
   - Bug in Non interacting code fixed (Now converges to required tolerance in real time)
   - iDEA_MB2 cleaned up

 * **v1.2.0** (2016-01-28)

   - MLP approximation added (constant f, 2 electron, time independent)
 * **v1.1.0** (2016-01-03)


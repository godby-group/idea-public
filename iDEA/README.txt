                *    ****     *****       *                 
                     *   *    *          * *                
                *    *    *   *         *   *               
                *    *     *  *****    *     *              
                *    *    *   *       *********             
                *    *   *    *      *         *            
                *    ****     ***** *           *           
                                                            
------------------------------------------------------------
|           Interacting Dynamic Electrons Approach         |
|              to Many-body Quantum Mechanics              |
|                                                          |
|                 Created by Piers Lillystone,             |                      
|                 Matt Hodgson, Jacob Chapman,             |
|               Thomas Durrant & Jack Wetherell            |
|                   The University of York                 |
------------------------------------------------------------

Contents
--------
1. Preparing for use
2. List of files
3. How to run

1. Preparing for use
--------------------
We need to update the bash.rc file to allow the intel compiler to be loacated.
Firstly locate your bash rc file:
	user@rwgu4:~$ cd
        user@rwgu4:~$ nano .bashrc
Add the following line below the line beginning # User specific aliases:
        . /phys/sfw/intel/current/bin/compilervars.sh intel64 # Intel compiler
Save the file using:
        ctrl-X, ENTER
Check the environment variable has been created using:
        user@rwgu4:~$ echo $MKLROOT
If no path is shown set the environment variable using:
        user@rwgu4:~$ export MKLROOT=/phys/sfw/intel/composer_xe_2013-09/composer_xe_2013_sp1.0.080/mkl
Then navigate to the directory with the iDEA code in.
Then compile the mkl file using:
        user@rwgu4:~/yourpath$ ./mkl.compiler
iDEA should be now ready to run, you should not have to complete these steps again.

2. List of files
----------------
antisym2e.py      - Constructs the expansion of reduction matrices for
                    the two electrons runs.
antisym3e.py      - Constructs the expansion of reduction matrices for
                    the three electrons runs. 
Density.py        - Calculates the electron density given the real time
                    many body wavefunction for two electron runs.
fort.so           - Contains the FORTRAN functions for antisymmetrisation
                    in the three electron runs.
iDEA.py           - The main program to run the iDEA code, running all the
                    programs specified by the parameters file.
iDEA_MB2.py       - Computes the many body wavefunction for two electron
                    runs.
iDEA_MB3.py       - Computes the many body wavefunction for three electron
                    runs.
iDEA_TDDFT2.py    - Reverse engineers the exact time dependent Kohn-Sham 
                    potential for a given non-equilibrium electron density
                    for 2 electrons.
iDEA_TDDFT3.py    - Reverse engineers the exact time dependent Kohn-Sham 
                    potential for a given non-equilibrium electron density
                    for 3 electrons.
mkl.compiler      - Used to compile the mkl.f file into a python callable
                    so file.
mkl.f             - Contains all the FORTRAN implemented parallelisation
                    techniques.
parameters.py     - The paramaters fiel used to set up the system and to 
                    tell iDEA.py what computations to perform.
plot.py           - Used to plot one timestep of the real time density for completed runs
animate.py        - Used to aniate the real time density for completed runs
Single-E....py    - Used to find the initial guess for the ground state 
                    wavefunction in three electron runs.
sprint.py         - Used to print data to the screen using the parameter
                    msglvl.

3. How to Run
-------------
1. Set the required grid parameters in parameters.py 
2. Set the required run parameters in parameters.py 
3. Run iDEA.py
4. Run animate.py to see time evolution of density

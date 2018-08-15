# iDEA

The interacting Dynamic Electrons Approach (iDEA) is a Python-Cython software
suite developed in Rex Godby's group at the University of York since
2010. It has a central role in a number of research projects related to
many-particle quantum mechanics for electrons in matter.

iDEA's main features are:

* Exact solution of the many-particle time-independent Schrödinger equation,
  including exact exchange and correlation
* Exact solution of the many-particle time-dependent Schrödinger equation,
  including exact exchange and correlation
* Simplicity achieved using spinless electrons in one dimension
* An arbitrary external potential that may be time-dependent
* Optimisation methods to determine the exact DFT/TDDFT Kohn-Sham potential
  and energy components
* Implementation of various approximate functionals (established and novel) for
  comparison
* Established and novel localisation measures
* Many-body perturbation theory with implementation of various approximate
  vertex corrections for comparison
  
## iDEA publications

A list of publications based on the iDEA code so far is available on [the iDEA code's home page](https://www-users.york.ac.uk/~rwg3/idea.html). 

## Citing iDEA

If you use iDEA, we would appreciate a reference to the iDEA code's home page, [https://www-users.york.ac.uk/~rwg3/idea.html](https://www-users.york.ac.uk/~rwg3/idea.html), and to one relevant publication from our group: you might consider: 

* For exact solution of the many-particle Schrödinger equation and reverse engineering of the exact DFT/TDDFT Kohn-Sham potential: [Publication 1](http://www-users.york.ac.uk/~rwg3/abst_81-110.html#Paper_87)
* For Hartree-Fock and hybrid calculations: [Publication 9](http://www-users.york.ac.uk/~rwg3/abst_81-110.html#Paper_97)
* For the iDEA code's local-density approximations: [Publication 10](http://www-users.york.ac.uk/~rwg3/abst_81-110.html#Paper_98)


## How to get iDEA

    git clone https://github.com/godby-group/idea-private.git iDEAL
    cd iDEAL
    pip install --user -e .[doc]
    python run.py # this runs an example system

## Documentation

The [iDEA documentation](https://www.cmt.york.ac.uk/group_info/idea_html/) is
hosted at the University of York.
Besides explaining the inner workings and theory behind iDEA, it includes
examples based on jupyter notebooks and pointers on 
[how to contribute](https://www.cmt.york.ac.uk/group_info/idea_html/dev/add.html) to the development of iDEA.

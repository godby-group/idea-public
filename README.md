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

## How to get iDEA

    git clone git@github.com:godby-group/idea-private.git
    cd idea-private
    pip install --user -e .[doc]
    python run.py # this runs an example system

## Documentation

The [iDEA documentation](https://www.cmt.york.ac.uk/group_info/idea_html/) is
hosted at the University of York.
Besides explaining the inner workings and theory behind iDEA, it includes
examples based on jupyter notebooks and pointers on 
[how to contribute](https://www.cmt.york.ac.uk/group_info/idea_html/dev/add.html) to the development of iDEA.
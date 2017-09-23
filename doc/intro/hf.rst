iDEA HF (Hartree-Fock)
======================


The HF code solves the Hartree-Fock equation to approximately calculate the ground state density for a one-dimensional finite system (using the softened Coulomb repulsion :math:`(|x-x'|+1)^{-1}`). A perturbing potential is then applied to the ground-state system and its evolution is calculated approximately through solving the time-dependent HF equation.

Calculating the ground-state
----------------------------

To compute the Hartree-Fock density of a system we fist begin with the non-interacting orbitals, as computed from the single-particle Schröedinger equation

.. math:: \{ \phi_{i},E_{i} \},

and from these the density from these orbitals :math:`n(x)`.

We then calculate the non-local exchange potential (Fock matrix) from these orbitals

.. math:: F(x,x') = \sum_{k} \psi_{k}(x) U(x,x') \psi_{k}(x'),

and the Hartree potential from the density:

.. math:: V_{H}(x) = \int n(x') U(x,x') dx'.

We then compute the Hamiltonian of the system:

.. math:: H(x,x') = K(x,x') + V_{ext}(x)\delta(x-x') + V_{H}(x)\delta(x-x') + F(x,x')

We can then find the eigenvalues and eigenfunctions of this Hamiltonian to obtain a new set of orbitals

.. math:: H\phi_{i} = \phi_{i}E_{i}.

From these orbitals we then repeat all of the above steps until the density reaches self-consistency.


Time-dependence
---------------

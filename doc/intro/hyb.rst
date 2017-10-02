iDEA HYB (Hybrid)
=================

The HYB code solves the Hybrid DFT equations to approximately calculate the ground state density for a one-dimensional finite system (using the softened Coulomb repulsion :math:`(|x-x'|+1)^{-1}`). A perturbing potential is then applied to the ground-state system and its evolution is calculated approximately through solving the time-dependent Hybrid DFT equation.

Calculating the ground-state
----------------------------

To compute the Hybrid density of a system we follow the same procedure as Hartree-Fock, but we perform a linear mixing of the Fock Operator and the LDA exchange-correlation potential
We first begin with the non-interacting orbitals, as computed from the single-particle Schrödinger equation:

.. math:: \{ \phi_{i},E_{i} \},

and from these the density from these orbitals :math:`n(x)`.

We then calculate the non-local exchange potential (Fock matrix) from these orbitals

.. math:: F(x,x') = \sum_{k} \phi_{k}(x) U(x,x') \phi^{*}_{k}(x'),

and the Hartree potential from the density:

.. math:: V_{H}(x) = \int n(x') U(x,x') dx'.

We then compute the Hamiltonian of the system:

.. math:: H(x,x') = K(x,x') + V_{ext}(x)\delta(x-x') + V_{H}(x)\delta(x-x') + \alpha*F(x,x') + (1-\alpha)V_{xc}^{LDA}

We can then find the eigenvalues and eigenfunctions of this Hamiltonian to obtain a new set of orbitals

.. math:: H\phi_{i} = \phi_{i}E_{i}.

From these orbitals we then repeat all of the above steps until the density reaches self-consistency.


Time-dependence
---------------
In order to evolve the Hartree-Fock density in time due to a perturbation we begin with the ground-state Hartree-Fock orbitals
as calculated above:

.. math:: \{ \phi_{i}\left( t=0\right),E_{i} \left( t=0\right) \},

from these we build the Time-dependence Hartree-Fock Hamiltonian:

.. math:: H(x,x') = K(x,x') + V_{ext}(x)\delta(x-x') + V_{ptrb}(x)\delta(x-x') + V_{H}(x)\delta(x-x') + \alpha*F(x,x') + (1-\alpha)V_{xc}^{LDA}

and using this Hamiltonian we use the Crank-Nicholson method to evolve the orbitals by one time-step:

.. math:: \{ \phi_{i}\left(t\right),E_{i} \left(t\right) \} \rightarrow \{ \phi_{i}\left(t=t+\Delta t\right),E_{i} \left(t=t+\Delta t\right) \}.

This precess repeats until the total time is reached, this gives us the time-dependent Hartree-Fock orbitals and density of the system.

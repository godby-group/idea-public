iDEA MLP (Mixed Localisation Potential)
=======================================

The mixed localisation potential (MLP) is an approximation to the Kohn-Sham (KS) potential of density functional theory (DFT). It is a combination of the familiar local density approximation (LDA) and the single orbital approximation (SOA); the latter is exact in the limit of complete electron localisation, whereas the former favours delocalised electrons. Hence, the MLP mixes the LDA and the SOA in proportions based of the degree of actual localisation in the system, as such [Hodgson2014] 

.. math:: V_{\mathrm{KS}} = (1- f)V^{\mathrm{LDA}}_{\mathrm{KS}} + f V^{\mathrm{SOA}}_{\mathrm{KS}}.

In principle :math:`f` depends on space, and for dynamic system also time. The form of the SOA KS potential is analytic (see below). Within the iDEA code, the MLP employs the the LDA of Ref. [Entwistle2016].

Calculating the ground state
----------------------------

In the ground state, the SOA KS potential has the following analytic form

.. math:: V^{\mathrm{SOA}}_{\mathrm{KS}} = \frac{\nabla^2 n}{4 n} - \frac{(\nabla n)^2}{8 n^2},

where :math:`n` is the electron density. The mixing term, :math:`f`, is approximated to be constant in space, and has been optimised based on a series on one-dimensional training systems [Torelli2018]

.. math:: f = \left | 1.49 \left \langle L \right \rangle - 0.984 \right |,

where :math:`\left \langle L \right \rangle` is the average value of the electron localisation function (ELF) of Ref. [Becke1990].

The MLP, when used to calculate the ground-state electron density, is fed into the KS equations until self-consistency is reached. The external potential is used as an initial guess for the KS potential, and thereafter the KS potential is mixed linearly per iteration of the self-consistent procedure. 

Time dependence
---------------

For time-dependent systems the form of the components of the MLP change; the LDA becomes the adiabatic LDA (ALDA), and the SOA becomes

.. math:: V^{\mathrm{SOA}}_{\mathrm{KS}} = \frac{\nabla^2 n}{4 n} - \frac{(\nabla n)^2}{8 n^2} - \frac{j}{2 n}

.. math:: A^{\mathrm{SOA}}_{\mathrm{KS}} = - \frac{j}{n},

where :math:`j` is the current density and :math:`A^{\mathrm{SOA}}_{\mathrm{KS}}` is the KS vector potential.

The mixing term, :math:`f`, can be used adiabatically, i.e., the electron density and orbitals at time :math:`t` are used in the ground-state form of :math:`f`, or fixed (:math:`f = f(0)`). 

The Crank-Nicolson method is used to propagate the KS equations through time after a perturbing field is applied to the system.

References
----------

.. [Hodgson2014]  M. J. P. Hodgson, J. D. Ramsden, T. R. Durrant and R. W. Godby, Physical Review B (Rapid Communications) 90 241107(R) (2014).

.. [Entwistle2016] M. T. Entwistle, M. J. P. Hodgson, J. Wetherell, B. Longstaff, J. D. Ramsden, and R. W. Godby, Phys. Rev. B 94, 205134 (2016).

.. [Torelli2018] D. Torelli, M. J. P. Hodgson and R. W. Godby, Unpublished.

.. [Becke1990] A. D. Becke and K. E. Edgecombe, J. Chem. Phys. 92, 5397–5403 (1990).

iDEA HFKS (Reverse-Engineering within Hartree-Fock-Kohn-Sham theory)
=============================
The HFKS code calculates the exact ground-state correlation potential :math:`V_{\mathrm{c}}(x)` within Hartree-Fock-Kohn-Sham (HFKS) theory for a given electron density :math:`n(x)`. 

Ground-state correlation potential
--------------------------------
The ground-state correlation potential :math:`V_{\mathrm{c}}(x)` is calculated by starting from a guess of zero and iteratively correcting using the algorithm:

.. math:: V_{\mathrm{c}}(x) \rightarrow V_{\mathrm{c}}(x) + \mu [n_{\mathrm{HFKS}}(x)^{p} - n(x)^{p}],

where :math:`n_{\mathrm{HFKS}}(x)` is the ground-state HFKS electron density, and :math:`\mu` and :math:`p` are convergence parameters. The correct :math:`V_{\mathrm{c}}(x)` is found when :math:`n_{\mathrm{HFKS}}(x) = n(x)`.

This correlation potential is added to the Fock exchange and external potential in the HFKS Hamiltonian. 

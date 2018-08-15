RE (Reverse-Engineering)
========================
The RE code calculates the exact Kohn-Sham (KS) potential [and hence exact exchange-correlation (xc) potential] for a given electron density :math:`n(x,t)`. 

Ground-state Kohn-Sham potential
--------------------------------
The ground-state KS potential :math:`V_{\mathrm{KS}}(x,0)` is calculated by starting from the unperturbed external potential and iteratively correcting using the algorithm:

.. math:: V_{\mathrm{KS}}(x,0) \rightarrow V_{\mathrm{KS}}(x,0) + \mu [n_{\mathrm{KS}}(x,0)^{p} - n(x,0)^{p}],

where :math:`n_{\mathrm{KS}}(x,0)` is the ground-state KS electron density, and :math:`\mu` and :math:`p` are convergence parameters. The correct :math:`V_{\mathrm{KS}}(x,0)` is found when :math:`n_{\mathrm{KS}}(x,0) = n(x,0)`.

Time-dependent Kohn-Sham potential
----------------------------------
The time-dependent KS potential :math:`V_{\mathrm{KS}}(x,t)` is calculated by applying a temporary gauge transformation and iteratively correcting a time-dependent KS vector potential :math:`A_{\mathrm{KS}}(x,t)`  using the algorithm:

.. math:: A_{\mathrm{KS}}(x,t) \rightarrow A_{\mathrm{KS}}(x,t) + \nu \bigg[ \frac{j_{\mathrm{KS}}(x,t) - j(x,t)}{n(x,t) + a} \bigg],

where :math:`j(x,t)` is the current density of the interacting system and :math:`j_{\mathrm{KS}}(x,t)` is the current density of the KS system. Once the correct :math:`A_{\mathrm{KS}}(x,t)` is found, the gauge transformation is removed to calculate the full time-dependent KS potential as a scalar quantity. 

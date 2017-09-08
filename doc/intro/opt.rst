iDEA OPT (Optimisation)
=======================

The OPT code calculates the exact external potential for a given ground-state electron density :math:`n_{\mathrm{target}}(x)`, with the system containing 1, 2 or 3 electrons. An initial guess for :math:`V_{\mathrm{ext}}(x)` is made and the electron density :math:`n(x)` is calculated using the EXT code. :math:`V_{\mathrm{ext}}(x)` is iteratively corrected using the algorithm:

.. math:: V_{\mathrm{ext}}(x) \rightarrow V_{\mathrm{ext}}(x) + \mu[n^{p}(x)-n_{\mathrm{target}}^{p}(x)],

where :math:`\mu` and :math:`p` are convergence parameters. The correct :math:`V_{\mathrm{ext}}(x)` is found when :math:`n(x) = n_{\mathrm{target}}(x)`.


iDEA OPT (Optimisation)
=======================

The OPT code calculates the exact external potential for a given electron density :math:`n_{\mathrm{target}}(x)`. An initial guess for :math:`V_{\mathrm{ext}}(x)` is made and it is iteratively corrected using the algorithm:

.. math:: V_{\mathrm{ext}}(x) \rightarrow V_{\mathrm{ext}}(x) + \mu[n^{p}(x)-n_{\mathrm{target}}^{p}(x)],

where :math:`\mu` and :math:`p` are convergence parameters. The correct :math:`V_{\mathrm{ext}}(x)` is found when :math:`n(x) = n_{\mathrm{target}}(x)`.


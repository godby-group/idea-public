*****************
Editing This Site
*****************

This documentation is written using `Sphinx <http://sphinx-doc.org>`_,
which employs the intuitive
`reStructuredText <http://sphinx-doc.org/rest.html#rst-primer>`_
format to generate HTML.
Below you'll find a few helpful examples -- for an extensive 
documentation of the format consult the resources linked above.

  * Write your mathematical formulae using LaTeX, 
    in line :math:`\exp(-i2\pi)` or displayed

    .. math:: f(x) = \int_0^\infty \exp\left(\frac{x^2}{2}\right)\,dx

  * You want to refer to a particular function or class? You can!

    .. autofunction:: iDEA.RE.GroundState
       :noindex:
  * Add images/plots to ``iDEA/doc/_static`` and then include them

    .. image:: ../_static/logo.png

  * Check out the source of any page via the link 
    in the bottom right corner.

|  

Once you are done editing the .rst files, do

.. code-block:: bash

    cd iDEA/doc
    make html # runs sphinx-build

|

Page source:

.. literalinclude:: edit.rst


********************
Adding documentation
********************

The documentation of iDEA consists of two main parts:

 * the :code:`doc/` folder, which contains documentation written in the intuitive
   `reStructuredText (reST) <http://sphinx-doc.org/rest.html#rst-primer>`_ format
 * `python docstrings <https://www.python.org/dev/peps/pep-0257/>`_ in the iDEA
   source code that document the functionality of its modules, classes, functions,
   etc. (specifically, we follow the
   `numpy convention <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_)

Both are important and you are most welcome to contribute to either of them.

In order to generate a browseable HTML version of the documentation (which is
displayed on the iDEA home page), we use a tool called
`Sphinx <http://sphinx-doc.org>`_, which also takes care of producing a
documentation of the source code (the API documentation).

Once you are done editing the documentation, produce an updated HTML version

.. code-block:: bash

    cd doc
    bash make_doc.sh 

See :ref:`generate-documentation` for details.


A short reST demo
------------------


  * Write your mathematical formulae using LaTeX, 
    in line :math:`\exp(-i2\pi)` or displayed

    .. math:: f(x) = \int_0^\infty \exp\left(\frac{x^2}{2}\right)\,dx

  * You want to refer to a particular function or class? You can!

    .. autofunction:: iDEA.RE.calculate_ground_state
       :noindex:
  * Add images/plots to ``iDEA/doc/_static`` and then include them

    .. image:: ../_static/logo.png

  * Check out the source of any page via the link 
    in the bottom right corner.

|  

reST source of this page:

.. literalinclude:: documentation.rst


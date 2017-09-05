Getting iDEA
============


Installation requirements
-------------------------

 * `Python <http://www.python.org>`_ 2.7 or later
 * `numpy <http://www.numpy.org>`_ 1.10 or later
 * `scipy <http://www.scipy.org>`_ 0.17 or later
 * `Cython <http://cython.org>`_ 0.22 or later
 * *(optional)* `matplotlib <http://matplotlib.org/>`_ 1.4 or later for post-processing

Installing iDEA
----------------

.. code-block:: bash
   
   # Clone from the central repository
   git clone USERNAME@rwgu4.its.york.ac.uk:/shared/storage/physrwg/trunk/iDEAL iDEAL

   # Install iDEA for your unix user, including documentation extension
   cd iDEAL
   pip install --user -e .[doc]

Updating iDEA
-------------

.. code-block:: bash

   # Pull all changes from central git repository
   git pull

.. _generate-documentation:

Generating the documentation
-----------------------------
A recent version of the documentation can be found on the iDEA web page.
If you are making changes to the code and/or the documentation, you may
need to generate the documentation by yourself

**Requirements**

 * `Sphinx <http://sphinx-doc.org>`_ 1.4 or later 
 * `numpydoc extension <https://pypi.python.org/pypi/numpydoc>`_ 0.7 or later (adds support for numpy-style docstrings)
 * `nbconvert extension <http://sphinx-doc.org>`_ 5.2 or later (renders static versions of jupyter notebooks)
 * (optional) `LaTeX <https://www.latex-project.org/get/>`_ (for the LaTeX version of the documentation)

Note: in order to install the required packages on a system without admin rights, do

.. code-block:: bash

   pip install --user sphinx numpydoc nbconvert

In order to produce the documentation in html and latex form:

.. code-block:: bash

   cd doc
   bash make_doc.sh
   # find html documentation in _build/html
   # find latex documentation in _build/latex 
   make latexpdf  # generates _build/latex/iDEA.pdf



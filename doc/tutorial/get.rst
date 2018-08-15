Getting iDEA
============


Installation requirements
-------------------------

 * `Python <http://www.python.org>`_ 3.3 or later
 * `numpy <http://www.numpy.org>`_ 1.10 or later
 * `scipy <http://www.scipy.org>`_ 0.17 or later
 * `Cython <http://cython.org>`_ 0.22 or later
 * *(optional)* `matplotlib <http://matplotlib.org/>`_ 1.4 or later for post-processing

Installing iDEA
----------------

.. code-block:: bash

   # Clone from the central repository
   git clone https://github.com/godby-group/idea-public.git idea-public

   # Install & compile iDEA for your unix user
   # (including packages for generating the documentation)
   cd idea-public
   pip install --user -e .[doc]

   # Run example calculation
   python run.py

Updating iDEA
-------------

.. code-block:: bash

   # Pull all changes from central git repository
   git pull origin master

   # Remove the compiled cython modules
   python setup.py clean --all

   # Recompile the cython modules
   python setup.py build_ext --inplace

.. _generate-documentation:

Generating the documentation
-----------------------------
A recent version of the documentation can be found on the iDEA web page.
If you are making changes to the code and/or the documentation, you may
need to generate the documentation by yourself:

.. code-block:: bash

   cd doc
   bash make_doc.sh
   # find html documentation in _build/html/index.html
   # find test coverage report in _build/coverage/index.html

Besides HTML, the iDEA documentation can also be compiled as a pdf.
If you have a LaTeX distribution installed on your system, simply do:

.. code-block:: bash

   cd doc
   make latexpdf

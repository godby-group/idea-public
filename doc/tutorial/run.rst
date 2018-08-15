Running iDEA
============

As it is a python package there are many different ways of running iDEA.

Using the iDEA code directly
----------------------------
Simply edit the parameters file :code:`parameters.py` and run

.. code-block:: bash

    python run.py

In order not to overwrite results from different calculations,
make sure to choose different run names for different inputs

.. literalinclude:: /../parameters.py
    :lines: 1-20
    :emphasize-lines: 9

Using the iDEA package in a python script
-----------------------------------------
Since iDEA is designed as a python package, it can be run from
everywhere, if you let your python installation know where the package is located.
During the installation of iDEA the `idea-public` directory should have been
added to PYTHONPATH. To test this has worked simply perform the following

.. code-block:: bash

    cd $test_folder                  # some folder you have created
    cp $path_to_iDEA/parameters.py . # where you have downloaded iDEA
    cp $path_to_iDEA/run.py .
    python run.py

Here, we are running iDEA much in the same way as before but your
:code:`$test_folder` can be located anywhere on your computer.

The main advantage of having an iDEA python package is that you can access its
functionality directly in a python script.

The example below uses a python loop to converge the grid spacing for an iDEA
calculation of a test system of non-interacting electrons in a harmonic well.

.. literalinclude:: /../examples/06_convergence/run.py

In order to run this example, do

.. code-block:: bash

    cd $path_to_iDEA/examples/06_convergence
    python run.py  # assuming you already added iDEA to your PYTHONPATH


Using the iDEA package in an ipython shell
------------------------------------------
As iDEA is a python package it can also be run from the interactive python shell
(:code:`ipython`). This is particularly useful as ipython has a convenient auto-complete
feature. The following example can be run in an ipython shell line-by-line.

.. code-block:: python

    from iDEA.input import Input

    # import parameters file
    pm = Input.from_python_file('parameters.py')

    # change parameters
    pm.run.EXT = True
    pm.run.NON = True
    print(pm.run)

    # run jobs
    results = pm.execute()

    # plot the relevant results
    x = pm.space.grid
    plt.plot(x, results.ext.gs_ext_den, label='exact')
    plt.plot(x, results.non.gs_non_den, label='non-interacting')
    plt.legend()
    plt.show()

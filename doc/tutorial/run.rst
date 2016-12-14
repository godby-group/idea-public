Running iDEA
============

iDEA is slowly evolving from a stand-alone python script to a python package.
For this reason there are different ways of running iDEA.

The old-fashioned way
---------------------
Simply edit the parameters file :code:`parameters.py` and run

.. code-block:: bash

    python iDEA.py

In order not to overwrite results from different calculations,
make sure to choose different run names for different inputs

.. literalinclude:: /../parameters.py
    :lines: 1-20
    :emphasize-lines: 7


Letting python know about iDEA
------------------------------
Since iDEA is designed as a python package, it can be run from
everywhere, if you let your python installation know where the package is located.

.. code-block:: bash

    export PYTHONPATH=$path_to_iDEA  # add this to your .bashrc
    cd $test_folder                  # some folder you have created
    cp $path_to_iDEA/parameters.py . # where you have downloaded iDEA
    cp $path_to_iDEA/iDEA.py .
    python iDEA.py

Here, we are running iDEA much in the same way as before but your
:code:`$test_folder` can be located anywhere on your computer.

Using iDEA package in a python script
-------------------------------------

The main advantage of having an iDEA python package is that you can access its
functionality directly in a python script.

The example below uses a python loop to converge the grid spacing for an iDEA
calculation of a test system of non-interacting electrons in a harmonic well.

.. literalinclude:: /../examples/ex02/run.py

In order to run this example, do

.. code-block:: bash

    cd $path_to_iDEA/examples/ex02
    python run.py  # assuming you already added iDEA to your PYTHONPATH




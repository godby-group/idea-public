Adding to iDEA
==============

The version control system used to manage the iDEA source code is 
`git <https://git-scm.com/>`_. Git offers a "learn git in 15 minutes" tutorial
found `here <https://try.github.io/>`_. 

Committing changes locally
--------------------------

Once you have made a change to a file, you will want to commit this change to your local repository. This
will ensure that as you pull changes from the central repository they will be automatically integrated into your work.
To see the list of files you have changed run

.. code-block:: bash

   git status

Then to add a file you want to commit run

.. code-block:: bash

   git add file_name

Once you have finished adding files you can commit your changes locally using

.. code-block:: bash

   git commit

You will be prompted to enter a commit message to describe your changes and save the file. Your changes are now committed!


Pushing your changes to the central repository
------------------------------------------------

Before asking for your changes to be included into iDEA, please make sure to
**create a unit test** that checks you feature is working as intended.

 * Naming convention: :code:`test_<your_module>.py`
 * start by copying a simple example, e.g. :code:`test_NON.py`
 * make sure your test is quick,
   it should run *in the blink of an eye*
   
At the **very minimum**

 1. Check that the existing unit tests aren't broken:

    .. code-block:: bash

       # run this in the base directory
       python -m unittest iDEA/test*.py


 2. Check that the documentation builds fine:

    .. code-block:: bash

       cd doc/
       bash make_doc.sh


To have the changes you have commited pulled into the central repository for
everyone to access, email jw1294@york.ac.uk with a pull request.

Advanced
.........

To check whether your code is properly covered by the unit tests, use the
`coverage module <http://coverage.readthedocs.io/>`_.

.. code-block:: bash

   # run this in the base directory
   coverage run -m iDEA/test*.py  # tests coverage
   coverage html  # generates report in doc/coverage/index.html                 


Pulling changes from the central repository
-------------------------------------------

Once another user has had their changes pulled into the central repository you will want to fetch 
these changes into your local repository and merge them with your work. To do this run

.. code-block:: bash

   git pull

You will not be able to perfrom this pull if you have untracked changes, you should first commit your changes as described above.
If you do not wish to commit the untracked changes and simply want to remove them run

.. code-block:: bash

   git stash

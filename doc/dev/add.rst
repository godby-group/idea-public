Adding to iDEA
==============

The iDEA source code is managed  using the `git <https://git-scm.com/>`_
version control system. Git offers a 
`"learn git in 15 minutes" tutorial <https://try.github.io/>`_. 

Committing changes locally
--------------------------

Before you start committing, make sure that your environment is properly configured:

.. code-block:: bash

   git config --global user.name "John Smith"
   git config --global user.email "john.smith@york.ac.uk"

Once you have made a set of changes you are happy with, you can commit the
change set to your local repository. This will ensure that as you pull changes
from the central repository they will be automatically integrated into your
work.
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


Contributing your changes to iDEA
---------------------------------

Before contributing your changes back to iDEA, make sure
to comply with our :doc:`best practises <practises>`.


 1. Fork the `iDEA repository on github <https://github.com/godby-group/idea-public>`_
 2. Assuming you've already committed your changes to a local repo, add a remote to your fork:

    .. code-block:: bash

       git remote add fork git@github.com:ltalirz/idea-public.git

 3. Pull from your fork (to synchronize), and then push local changes back to github

    .. code-block:: bash

       git pull fork master # may be asked to merge
       git push fork master

 4. `Create a pull request <https://github.com/godby-group/idea-public/pulls>`_ from your fork to the iDEA repo on github (you'll need to click on *compare across forks*)

Once a core developer maintainer has reviewed your pull request, your changes
will be incorporated into iDEA.

Pulling the latest changes from iDEA
------------------------------------

As the development of iDEA is progressing, you'll need to update your fork and
your local repository from time to time.

Updating your local repository:

.. code-block:: bash

   git remote add upstream git@github.com:godby-group/idea-public.git
   git pull upstream master

After this, also update your fork on github:

.. code-block:: bash

   git push fork master

**Note:** You will not be able to perfrom this pull if you have untracked changes, you should first commit your changes as described above.
If you do not wish to commit the untracked changes and simply want to remove them run

.. code-block:: bash

   git stash

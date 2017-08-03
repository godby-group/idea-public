Reverse engineering
-------------------

One of the great abilities of iDEA is being able to obtain the
:math:`\textit{exact}` KS potential. This is done by solving the
Schrodinger equation exactly, obtaining the exact electron density. From
here, the iDEA code then works backwards to see what KS potential would
have given this electron density (full details of the method are in the
manual written by Mike Entwistle). The reason for doing this is that it
allows us to compare the exact reverse-engineered KS potential with that
given by approximations, such as the LDA - using this comparison to
improve the approximations we're using.


#!/bin/bash

echo "### Symlinking Jupyter Notebooks ###"
src=../../examples
dest=examples
ln -sfn $src/01_get_started_basics/get_started_basics.ipynb $dest
ln -sfn $src/02_get_started_further/get_started_further.ipynb $dest
ln -sfn $src/03_well/well.ipynb $dest
ln -sfn $src/04_double_well/double_well.ipynb $dest
ln -sfn $src/05_tunneling/tunneling.ipynb $dest
ln -sfn $src/06_convergence/convergence.ipynb $dest

echo "### Preparing API documentation tree ###"
fldr=apidoc
mkdir -p $fldr
rm $fldr/*rst
sphinx-apidoc -o $fldr ../iDEA

rm -fr _build
echo "### Making HTML documentation ###"
make html
echo "### Preparing LaTeX documentation ###"
make latex

echo "### Producing test coverage report ###"
coverage run -m unittest discover ..
coverage html

echo "### Find HTML documentation in _build/html/index.html ###"
echo "### Find LaTeX documentation in _build/latex (type 'make') ###"
echo "### Find test coverage report in _build/coverage/index.html ###"

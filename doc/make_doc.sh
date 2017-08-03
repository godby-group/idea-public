#!/bin/bash

echo "### symlinking Jupyter Notebooks ###"
src=../../jupyter
dest=jupyter
#ln -sfn $src jupyter/notebooks
ln -sfn $src/00_get_started/get_started.ipynb $dest/
ln -sfn $src/01_well/well.ipynb $dest/
ln -sfn $src/02_double_well/double_well.ipynb $dest/
ln -sfn $src/03_tunneling/tunneling_solution.ipynb $dest/
#for f in "$jupyter_fldr/*.ipynb"; do
#    jupyter nbconvert --to html --output-dir tutorial  $f 
#done


echo "### Preparing API documentation tree ###"
fldr=apidoc
mkdir -p $fldr
rm $fldr/*rst
sphinx-apidoc -o $fldr ../iDEA

echo "### Making documentation ###"
rm -fr _build
make html

echo "### Please find the documentation website in _build/html ###"

#!/bin/bash

#echo "### Converting Jupyter Notebooks ###"
jupyter_fldr=../../jupyter
ln -sfn ${jupyter_fldr}/01_well/well.ipynb jupyter/
ln -sfn ${jupyter_fldr}/02_double_well/double_well.ipynb jupyter/
ln -sfn ${jupyter_fldr}/03_tunneling/tunneling_solution.ipynb jupyter/
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


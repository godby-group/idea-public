#!/bin/bash
fldr=apidoc
mkdir -p $fldr
rm $fldr/*rst
sphinx-apidoc -o $fldr ../iDEA

rm -fr _build
make html

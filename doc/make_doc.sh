#!/bin/bash
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

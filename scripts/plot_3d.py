#!/usr/bin/env python
"""Plot a complex quantity M[r,r',tau]

The data is read from a pickle file.
The plot is saved as a movie
"""
import argparse
import sys
import os
import pickle
import re

import iDEA.plot as iplt
from iDEA.input import Input

parser = argparse.ArgumentParser(
    description="Plots a complex quantity F[r,r',tau] into a movie")
parser.add_argument('--version', action='version', version='%(prog)s 25.01.2017')
parser.add_argument(
    'quantities',
    nargs='+',
    metavar='NAME',
    help='Name of the quantity to plot. Multiple quantities should be separated\
    by spaces.')
parser.add_argument(
    '--parameters',
    metavar='FILENAME',
    default='parameters.py',
    help='Name of the quantity to plot.')
parser.add_argument(
    '--format',
    metavar='STRING',
    default='mp4',
    help='Output format: "png", "pdf" or "mp4"')
parser.add_argument(
    '--stride',
    metavar='INT',
    default=1,
    help='Plot every STRIDE frames')
parser.add_argument(
    '--final',
    metavar='INT',
    default=0,   # If 0 will be replace with tau_n
    help='Final frame to plot')
args = parser.parse_args()

pm = Input.from_python_file(args.parameters)

for name in args.quantities:

    pfile = "raw/{}.db".format(name)
    print("Reading {}".format(pfile))
    data = pickle.load( open(pfile, "rb") )

    if re.search('_it',name):
        space = 'it'
    elif re.search('_iw', name):
        space = 'iw'
    else:
        space = 'it'

    iplt.plot3d(data, name, pm, space, args.format, step=args.stride, final=args.final)

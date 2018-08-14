#!/usr/bin/env python
import pickle

import numpy as np


def optimize():
   """Calculate optimal alpha for hybrid Hamiltonian.
   
   Return a tuple of three optimal values of alpha, corresponding to
   each condition:
   (LUMO-A, LUMO-HOMO, LUMO-I).
   If no alpha satisfies a condition, return None.
   """
   # Load the raw data.
   filenames = ['alphas', 'enN', 'enNm', 'eigH', 'eigL']
   data = {}
   print('Reading raw data:')
   for fn in filenames:
      print('gs_hyb_{}.db'.format(fn))
      with open('raw/gs_hyb_{}.db'.format(fn), 'rb') as f:
         data[fn] = pickle.load(f)

   # Ensure all data has the same shape.
   for i in data.values():
      for j in data.values():
         if np.any(i != j) and i.shape != j.shape:
            raise IOError('all files must have same shape of data.')

   # Create arrays for which to find zero position.
   lumo_a = data['enN'] - data['enNm'] - data['eigL']
   lumo_homo = data['eigH'] - data['eigL']
   homo_i = data['enN'] - data['enNm'] - data['eigH']

   # Calculate optimal alphas.
   optimals = []
   for y_values in [lumo_a, lumo_homo, homo_i]:
      alphas = data['alphas']
      diff = np.diff(y_values)
      # Reverse arrays if sorted in descending order.
      if np.all(diff < 0):
         alphas = alphas[::-1]
         y_values = y_values[::-1]
      elif not np.all(diff > 0):
         print('Unsorted array for condition. Returning None.')
         optimals.append(None)
         continue
      index_upper = np.searchsorted(y_values, 0)
      if 0 < index_upper < y_values.shape[0]:
         index_lower = index_upper - 1
         y_upper = y_values[index_upper]
         y_lower = y_values[index_lower]
         y_frac = y_lower / (y_lower - y_upper)
         root = (1 - y_frac) * alphas[index_lower] + y_frac * alphas[index_upper]
      elif y_values[0] == 0:
         root = alphas[0]
      else:
         root = None
      optimals.append(root)
   if all(a is None for a in optimals):
      print("No optimal alpha found using any condition. Try increasing the input range of alpha.")
   return tuple(optimals)


if __name__ == '__main__':
   optimal_alphas = optimize()
   cond_names = ("LUMO-A", "LUMO-HOMO", "HOMO-I (GKT)")
   print('\nOptimal alpha values:')
   for index, item in enumerate(optimal_alphas):
      if item is not None:
         print("{:12}: {}".format(cond_names[index], item))

""" Tests for the result class

""" 
from __future__ import absolute_import
from . import results
from . import input
import unittest
import numpy as np
import numpy.testing as nt

# decimal places for comparison of results
d = 6

class resultsTest(unittest.TestCase):
    """ Tests results object

    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        pm.run.name = 'unittest'
        self.pm = pm

    def test_save_1(self):
        r""" Checks that saving works as expected
        
        """
        pm = self.pm
        r = results.Results()

        data = np.zeros(10)
        r.add(data, "first_data")
        nt.assert_array_equal(r.first_data, data)


        self.assertEqual(r._not_saved,["first_data"])
        # here, one would normally do 'r.save(pm)'
        # but we don't want to pollute the filesystem...
        r._saved.add("first_data")

        r.add(data, "second_data")
        self.assertEqual(r._not_saved,["second_data"])

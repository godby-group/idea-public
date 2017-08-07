""" Tests for the result class

""" 
from __future__ import absolute_import
from . import results
from . import input
import unittest
import numpy as np

# decimal places for comparison of results
d = 6

class resultsTest(unittest.TestCase):
    """ Tests results object

    """ 

    def setUp(self):
        """ Sets up harmonic oscillator system """
        pm = input.Input()
        self.pm = pm

    def test_save_1(self):
        r""" Checks that saving works as expected
        
        """
        pm = self.pm
        r = results.Results()

        data = np.zeros(10)
        r.add(data, "first_data")

        self.assertEqual(r.__to_save__,["first_data"])
        # this would normally be handled by
        #r.save(pm)
        # but we don't want to pollute the filesystem...
        r.__saved__.append("first_data")

        r.add(data, "second_data")
        self.assertEqual(r.__to_save__,["second_data"])

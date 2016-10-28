"""Bundles and saves iDEA results

"""
import numpy as np
import pickle
import copy as cp
import sprint

class Results(object):
    """Container for results.

    At the moment, this class is simply a convenient container for the results
    of a calculation. Additional functionality may be added at a later point.
    """
    method_dict = {
        'non': 'non-interacting',
        'ext': 'exact',
        'hf': 'Hartree-Fock',
        'lda': 'LDA',
    }

    quantity_dict = {
        'den': r'$\rho$',
        'vxt': r'$V_{ext}$',
        'vh': r'$V_{H}$',
        'vxc': r'$V_{xc}$',
        'vks': r'$V_{KS}$',
    }

    @staticmethod
    def label(shortname):
        """returns full label for shortname of result.

        This refers to shortnames used for 1d quantities
        saved by iDEA.
        E.g. 'non_den' => r'non-interacting $\rho$'
        """
        m, l = shortname.split('_')
        s  = "{} ({})".format(Results.quantity_dict[l], Results.method_dict[m])

        return s

    def add(self,results,name):
        """Add results to the container."""

        ## If we are adding another Results instance, copy attributes
        #if isinstance(results,Results):
        #    self.__dict__.update(results.__dict__)
        #else:
        setattr(self, name, cp.deepcopy(results))

    def save(self,dir,verbosity='default'):
        """Save results to disk."""
        for key,val in self.__dict__.iteritems():
            if isinstance(val,Results):
                val.save(dir)
            else:
                outname = "{}/{}.db".format(dir,key)
                if verbosity:
                    sprint.sprint("Saving {} to {}".format(key,outname),0,verbosity)    
                else:
                    sprint.sprint("Saving {} to {}".format(key,outname),0,verbosity)
                f = open(outname, 'wb')
                pickle.dump(val,f)
                f.close()
                #np.savetxt(outname, val)

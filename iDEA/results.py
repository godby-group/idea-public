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
        r""" returns full label for shortname of result.

        Expand shortname used for 1d quantities saved by iDEA.
        E.g. 'non_den' => 'non-interacting $\rho$'
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

        if not hasattr(self, name) or not isinstance(results,Results):
            setattr(self, name, cp.deepcopy(results))
        # if name already exists and we are adding another Results instance
        # copy its attributes
        else:
            getattr(self, name).__dict__.update(results.__dict__)


    def read(self, name, dir, verbosity='default'):
        """Read results from disk.

        Results are both added to results object *and* returned.

        parameters
        ----------
        name : string
            name of results to be read (filename = name.db)
        dir : string
            directory where to read results from, relative to cwd
        verbosity : string
            additional info will be printed for verbosity 'high'

        Returns l
        """
        filename = "{}/{}.db".format(dir,name)
        sprint.sprint("Saving {} to {}".format(filename,dir),0,verbosity)
        f = open(filename, 'rb')
        data = pickle.load(f)
        f.close()
        
        setattr(self, name, data)
        return data



    def save(self, dir, verbosity='default', list=None):
        """Save results to disk.

        parameters
        ----------
        dir : string
            directory where to save results, relative to cwd
        verbosity : string
            additional info will be printed for verbosity 'high'
        list : array_like
            if set, only the listed results will be saved
        """
        for key,val in self.__dict__.iteritems():
            if list is None or key in list:
                if isinstance(val,Results):
                    val.save(dir)
                else:
                    outname = "{}/{}.db".format(dir,key)
                    sprint.sprint("Saving {} to {}".format(key,outname),0,verbosity)    
                    f = open(outname, 'wb')
                    pickle.dump(val,f)
                    f.close()
                    #np.savetxt(outname, val)

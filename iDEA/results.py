"""Bundles and saves iDEA results

"""
import numpy as np
import pickle
import copy as cp

class Results(object):
    """Container for results.

    A convenient container for storing, reading and saving the results of a
    calculation.
    """
    calc_dict = {
        'td': 'time-dependent',
        'gs': 'ground state',
    }

    method_dict = {
        'non': 'non-interacting',
        'ext': 'exact',
        'hf': 'Hartree-Fock',
        'lda': 'LDA',
    }

    quantity_dict = {
        'den': r'$\rho$',
        'cur': r'$j$',
        'eigf': r'$\psi_j$',
        'eigv': r'$\varepsilon_j$',
        'vxt': r'$V_{ext}$',
        'vh': r'$V_{H}$',
        'vxc': r'$V_{xc}$',
        'vks': r'$V_{KS}$',
    }

    @staticmethod
    def label(shortname):
        r""" returns full label for shortname of result.

        Expand shortname used for 1d quantities saved by iDEA.
        E.g. 'gs_non_den' => 'ground state $\rho$ (non-interacting)'
        """
        c, m, q = shortname.split('_')
        s  = r"{} {} ({})".format(Results.calc_dict[c], Results.quantity_dict[q], Results.method_dict[m])

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

    @staticmethod
    def read(name, pm, dir=None):
        """Reads and returns results from pickle file

        parameters
        ----------
        name : string
            name of result to be read (filepath = raw/name.db)
        pm : object
            iDEA.input.Input object
        dir : string
            directory where result is stored
            default: pm.output_dir + '/raw'

        Returns data
        """
        if dir is None:
            dir = pm.output_dir + '/raw'

        filename = "{}/{}.db".format(dir,name)
        pm.sprint("Reading {} from {}".format(name,filename),0)
        #pm.sprint("Reading {} from {}".format(Results.label(name),filename),0)
        f = open(filename, 'rb')
        data = pickle.load(f)
        f.close()
        
        return data

    def add_pickled_data(self, name, pm, dir=None):
        """Read results from pickle file and adds to results.

        parameters
        ----------
        name : string
            name of results to be read (filepath = raw/name.db)
        pm : object
            iDEA.input.Input object
        dir : string
            directory where result is stored
            default: pm.output_dir + '/raw'
        """
        data = self.read(name, dir, verbosity)
        setattr(self, name, data)



    def save(self, pm, dir=None, list=None):
        """Save results to disk.

        parameters
        ----------
        pm : object
            iDEA.input.Input object
        dir : string
            directory where to save results
            default: pm.output_dir + '/raw'
        verbosity : string
            additional info will be printed for verbosity 'high'
        list : array_like
            if set, only the listed results will be saved
        """
        if dir is None:
            dir = pm.output_dir + '/raw'

        for key,val in self.__dict__.iteritems():
            if list is None or key in list:
                if isinstance(val,Results):
                    val.save(pm, dir)
                else:
                    outname = "{}/{}.db".format(dir,key)
                    pm.sprint("Saving {} to {}".format(key,outname),0)
                    f = open(outname, 'wb')
                    pickle.dump(val,f)
                    f.close()
                    #np.savetxt(outname, val)

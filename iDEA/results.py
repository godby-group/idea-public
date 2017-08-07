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
        'tden' : r'$\rho$',
        'W' : r'$W$',
        'S' : r'$\Sigma$',
        'Sc' : r'$\Sigma_{c}$',
        'Sx' : r'$\Sigma_{x}$',
        'Sxc' : r'$\Sigma_{xc}$',
    }

    def __init__(self):
        self.__saved__ = []  # list of results saved to disk

    @property
    def __to_save__(self):
        """List of results not yet saved to disk"""
        results = list(self.__dict__.keys())
        if '__saved__' in results: results.remove('__saved__')

        return [r for r in results if r not in self.__saved__]

    @staticmethod
    def label(shortname):
        r""" returns full label for shortname of result.

        Expand shortname used for quantities saved by iDEA.
        E.g. 'gs_non_den' => 'ground state $\rho$ (non-interacting)'
        """
        c, m, q = shortname.split('_')
        s  = r"{} {} ({})".format(Results.calc_dict[c], Results.quantity_dict[q], Results.method_dict[m])

        return s

    def add(self,results,name):
        """Add results to the container.
        """

        if hasattr(self, name):
            if name in self.__saved__:
                self.__saved__.remove(name)

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
        data = pickle.load(f, encoding='latin1')
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
        data = self.read(name, pm, dir)
        self.add(data, name)


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
            if set, saves listed results
            if not set, saves results that haven't been saved before
        """
        if dir is None:
            dir = pm.output_dir + '/raw'

        if list:
            to_save = list
        else:
            to_save = self.__to_save__

        for key,val in self.__dict__.items():
            if key in to_save:
                if isinstance(val,Results):
                    val.save(pm, dir)
                else:
                    outname = "{}/{}.db".format(dir,key)
                    pm.sprint("Saving {} to {}".format(key,outname),0)
                    f = open(outname, 'wb')
                    pickle.dump(val,f,protocol=4) # protocol=4 for large files (<4GB)
                    f.close()
                    #np.savetxt(outname, val)
                if key not in self.__saved__:
                    self.__saved__.append(key)

    def save_hdf5(self, pm, dir=None, list=None, f=None):
        """Save results to HDF5 database.

        This requires the h5py python package.

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
        f : HDF5 handle
            handle of HDF5 file (or group) to write to
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("Need hd5py package for saving results in HDF5 format.")
        if dir is None:
            dir = pm.output_dir + '/raw'
        outname = "{}/store.hdf5".format(dir)

        openfile = False
        if f is None:
            f = h5py.File(outname, "a")
            openfile = True

        for key,val in self.__dict__.items():
            if list is None or key in list:
                if isinstance(val, Results):
                    try:
                        grp = f.create_group(key)
                    except ValueError:
                        grp = f[key]
                    val.save_hdf5(pm, dir, f=grp)
                #elif isinstance(val, np.ndarray):
                #    f.create_dataset(key, data=val)
                else:
                    if isinstance(val, np.ndarray):
                        compression = 'lzf'
                    else:
                        compression = None

                    if key not in list(f.keys()):
                        pm.sprint("Saving {} to {}".format(key,outname),0)
                        f.create_dataset(key, data=val, compression=compression)
                    else:
                        # Note: deleting data in hdf5 is not guaranteed to
                        # free the corresponding disk space
                        pm.sprint("Overwriting {} in {}. This can waste disk space.".format(key,outname),0)
                        del f[key]
                        f.create_dataset(key, data=val, compression=compression)
        if openfile:
            f.close()

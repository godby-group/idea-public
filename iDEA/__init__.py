"""interacting Dynamic Electrons Approach (iDEA)

The iDEA code allows to propagate the time-dependent Schroedinger equation for
2-3 electrons in one-dimensional real space.
Compared to other models, such as the Anderson impurity model, this allows us
to treat exchange and correlation throughout the system and provides additional
flexibility in bridging the gap between model systems and ab initio
descriptions.
"""
make_fortran = True

if make_fortran:
    # make Fortran libraries
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import subprocess
    # note: this could be made more clever to
    #   automatically detect different environments
    #msg = subprocess.check_output(["make"], cwd=dir_path)
    
    # First check whether anything needs to be made (needed to print a message
    # *only* if we are actually making something)
    p = subprocess.Popen(["make","--just-print"], cwd=dir_path, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if out.find("Nothing to be done") != -1:
        pass
    else:
        print("Compiling Fortran libraries...")
        p = subprocess.Popen(["make"], cwd=dir_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if err:
            print(err)
            raise ImportError("Error while compiling Fortran libraries. Try typing 'make' in iDEA subdirectory.".format(err))

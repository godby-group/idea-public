from setuptools import setup
from distutils.util import convert_path
from distutils.extension import Extension
from distutils.command.clean import clean
from Cython.Build import cythonize
import os
import platform
import numpy

# read package info
package_name = "iDEA"
info_dict = {}
with open(convert_path('{}/info.py'.format(package_name))) as f:
    exec(f.read(), info_dict)


# define custom clean command
arch = platform.system()
class clean_inplace(clean):
    """Clean shared libararies"""

    # Calls the default run command, then deletes .so files
    def run(self):
        clean.run(self)
        files=os.listdir(package_name)

        if arch == 'Linux':
            exts = ('.so')
        elif arch == 'Darwin':
            exts = ('.so', '.dSYM')
        elif arch == 'Windows':
            exts = ('.dll')
        else:
            exts = ('.so')

        for f in files:
            if f.endswith(exts):
                path = os.path.join(package_name,f)
                print("Removing {}".format(path))
                os.remove(path)


extensions = [ 
    Extension("*", ["{}/*.pyx".format(package_name)], include_dirs = [numpy.get_include()]) 
]

setup(
    name='iDEA',
    packages=[package_name],
    description = 'interacting Dynamic Electrons Approach',
    author = info_dict['authors_short'],
    license = 'MIT',
    version = info_dict['release'],
    url = 'http://www.cmt.york.ac.uk/group_info/idea_html/',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    include_package_data=True,
    install_requires=[
        'matplotlib>=1.4',
        'numpy>=1.10',
        'scipy>=0.17',
        'cython>=0.22',
    ],
    extras_require = {
    'doc':  ['sphinx>=1.4', 'numpydoc', 'jupyter','nbsphinx', 'coverage'],
    },
    ext_modules = cythonize(extensions),
    cmdclass = {'clean': clean_inplace},
)

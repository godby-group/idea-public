from setuptools import setup
from distutils.util import convert_path
from Cython.Build import cythonize

package_name = "iDEA"
info_dict = {}
with open(convert_path('{}/info.py'.format(package_name))) as f:
    exec(f.read(), info_dict)


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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
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
    'doc':  ['sphinx>=1.4', 'numpydoc', 'jupyter','nbsphinx'],
    },
    ext_modules = cythonize([
        "{}/construct_response_function.pyx".format(package_name),
    ]),
)

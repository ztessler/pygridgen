"""Gridgen is a tool for creating curvilinear grids.

Requires:
    numpy
    matplotlib
    pyproj or basemap (optional)
"""

from setuptools import setup, find_packages

classifiers = """\
Development Status :: beta
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: "BSD 3-clause"
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
"""


doclines = __doc__.split("\n")

setup(
    name="pygridgen",
    version='0.2.dev',
    description=doclines[0],
    long_description="\n".join(doclines),
    author="Robert Hetland",
    author_email="hetland@tamu.edu",
    url="http://github.com/hetland/pygridgen",
    packages=find_packages(exclude=[]),
    license="BSD 3-clause",
    platforms="Python 2.7, 3.4, 3.5 and later.",
    ext_package='pygridgen',
    classifiers=classifiers.split("\n"),
    install_requires=['numpy', 'matplotlib'],
)


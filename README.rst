`pygridgen`
===========
.. image:: https://travis-ci.org/phobson/pygridgen.svg?branch=master
    :target: https://travis-ci.org/phobson/pygridgen
.. image:: https://coveralls.io/repos/phobson/pygridgen/badge.svg?branch=use-properties-CGrid&service=github
  :target: https://coveralls.io/github/phobson/pygridgen?branch=use-properties-CGrid


A Python interface to Pavel Sakov's `gridgen`_ library

.. _gridgen: https://github.com/sakov/gridgen-c

The full documentation for this for library is `here`_.

.. _here: https://phobson.github.io/pygridgen


Installation
------------
For linux and Mac OS X, binaries are available trouth the IOOS conda `channel`_.
Installation is as simple as::

    $ conda create --name=grid python=3.4
    $ source activate grid
    $ conda install --channel=IOOS pygridgen

.. _channel: http://anaconda.org/IOOS/pygridgen

To build from source, see the installation guides for your platform:

 + `Building on OS X`_
 + `Building on Linux`_

.. _Building on OS X: install/build_osx.rst
.. _Building on Linux: install/build_linux.rst

Python Dependencies
-------------------

Basics
~~~~~~

Provided that all of the shared C libraries are installs as covered in `Installation`_, the remaining python depedencies are the following::

  * numpy
  * matplotlib
  * pyproj or mpl-basemap (only if working with geographic coordinates)

Testing
~~~~~~~

Tests are written using the `nose` package.
From the source tree, run them simply with by invoking `nosetests` in a terminal.

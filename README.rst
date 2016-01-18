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

For more detailed documentation on grid generation, manipulation, and visualization,
see the documentation for `pygridtools`_.

.. _pygridtools: https://phobson.github.io/pygridtools


Credits
-------
This fork of ``pygridgen`` stands on the very tall shoulders of `Robert Hetland`_ of Texam A&M Universits.
Many thanks to him, `Richard Signell`_, `Fillipe Fernandes`_, and all of the other contributors.

.. _Robert Hetland: https://github.com/hetland
.. _Richard Signell: https://github.com/rsignell-usgs
.. _Fillipe Fernandes: https://github.com/ocefpaf


Python Dependencies
-------------------

Basics
~~~~~~

Provided that all of the shared C libraries are installed as covered in per the instructions below, the remaining python depedencies are the following:

* numpy
* matplotlib
* pyproj (only if working with geographic coordinates)

Testing
~~~~~~~

Tests are written using the `nose` package.
From the source tree, run them simply with by invoking ``nosetests`` in a terminal.

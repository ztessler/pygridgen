# encoding: utf-8
'''Tools for creating curvilinear grids using gridgen by Pavel Sakov'''
__docformat__ = "restructuredtext en"

import ctypes
import numpy as np
import pdb

xbry = np.array([0.9, 0.1, 0.2, 0.9])
ybry = np.array([0.9, 0.9, 0.1, 0.1])
beta = np.array([1.0, 1.0, 1.0, 1.0])

nx = 30
ny = 30
ul_idx = 0

nnodes=14
precision=1.0e-12
nppe=3
newton=True
thin=True
checksimplepoly=True
verbose=True

_libgridgen = np.ctypeslib.load_library('libgridgen', '/usr/local/lib')

print _libgridgen

_libgridgen.gridgen_generategrid2.restype = ctypes.c_void_p
_libgridgen.gridnodes_getx.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
_libgridgen.gridnodes_gety.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
_libgridgen.gridnodes_getnce1.restype = ctypes.c_int
_libgridgen.gridnodes_getnce2.restype = ctypes.c_int
_libgridgen.gridmap_build.restype = ctypes.c_void_p


nbry = len(xbry)

nsigmas = ctypes.c_int(0)
sigmas = ctypes.c_void_p(0)
nrect = ctypes.c_int(0)
xrect =  ctypes.c_void_p(0)
yrect = ctypes.c_void_p(0)
        
ngrid = ctypes.c_int(0)
xgrid = ctypes.POINTER(ctypes.c_double)()
ygrid = ctypes.POINTER(ctypes.c_double)()
        
_gn = _libgridgen.gridgen_generategrid2(
     ctypes.c_int(nbry), 
     (ctypes.c_double * nbry)(*xbry), 
     (ctypes.c_double * nbry)(*ybry), 
     (ctypes.c_double * nbry)(*beta),
     ctypes.c_int(ul_idx), 
     ctypes.c_int(nx), 
     ctypes.c_int(ny), 
     ngrid, 
     xgrid, 
     ygrid,
     ctypes.c_int(nnodes), 
     ctypes.c_int(newton), 
     ctypes.c_double(precision),
     ctypes.c_int(checksimplepoly), 
     ctypes.c_int(thin), 
     ctypes.c_int(nppe),
     ctypes.c_int(verbose),
     ctypes.byref(nsigmas), 
     ctypes.byref(sigmas), 
     ctypes.byref(nrect),
     ctypes.byref(xrect), 
     ctypes.byref(yrect) )


# print 'run getx'
# x = _libgridgen.gridnodes_getx(_gn)
#
# print 'reshape result.'
# x = np.asarray([x[0][i] for i in range(ny*nx)])
# x.shape = (ny, nx)


print 'run gety'
y = _libgridgen.gridnodes_gety(_gn)

print 'reshape result.'
y = np.asarray([y[0][i] for i in range(ny*nx)])
y.shape = (ny, nx)


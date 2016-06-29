import sys
import matplotlib
matplotlib.use('agg')

import pygridgen
status = pygridgen.test(*sys.argv[1:])
sys.exit(status)
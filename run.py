__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.tools.io import c_print
from adaptiveRank.arm import Bernoulli

from optparse import OptionParser

#====================
# RUNNING PARAMETERS
#====================
parser = OptionParser(usage="usage: %prog [options]",
        version="%prog 1.0")
parser.add_option('--gamma', dest = 'gamma', default = 0.5, type = "float", help = "Discount parameter")
parser.add_option('--N', dest = 'N', default = 100, type = "int", help = "Time horizon")
parser.add_option('--k', dest = 'K', default = 10, type = "int", help = "Number of arms")
parser.add_option('--nrep', dest = 'nrep', default = 1, type = "int", help = "Number of repetitions")
(opts, args) = parser.parse_args()

gamma = opts.gamma
N = opts.N
k = opts.K
nrep = opts.nrep

c_print(2, "run N {} k {} nrep {}".format(N,k,nrep))

#=====================
# INITIALIZATION 
#====================


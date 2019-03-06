__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.tools.io import c_print
from adaptiveRank.arm import Bernoulli
from adaptiveRank.Evaluation import Evaluation
from adaptiveRank.environment import MAB

from optparse import OptionParser
from numpy import mean, std, zeros
from joblib import Parallel, delayed

#====================
# RUNNING PARAMETERS
#====================
parser = OptionParser(usage="usage: %prog [options]",
        version="%prog 1.0")
parser.add_option('--gamma', dest = 'GAMMA', default = 0.5, type = "float", help = "Discount parameter")
parser.add_option('--max_delay', dest = 'MAX_DELAY', default = 5, type = "int", help = "Memory size for the discount")
parser.add_option('--N', dest = 'N', default = 100, type = "int", help = "Time horizon")
parser.add_option('--k', dest = 'K', default = 10, type = "int", help = "Number of arms")
parser.add_option('--nrep', dest = 'NREP', default = 1, type = "int", help = "Number of repetitions")
(opts, args) = parser.parse_args()

GAMMA = opts.GAMMA
MAX_DELAY = opts.MAX_DELAY
HORIZON = opts.N
N_ARMS = opts.K
N_REPETITIONS = opts.NREP

c_print(2, "run horizon: {} number of arms: {} number repetitions: {}".format(HORIZON, N_ARMS, N_REPETITIONS))

#=====================
# INITIALIZATION 
#====================
mab = MAB(HORIZON, N_ARMS, N_REPETITIONS, GAMMA, MAX_DELAY)
policies = []
policies_name = ['Random', 'Combinatorial Random']
#assert len(policies) == len(policies_name), "Check coherence policies with their names"
N_POLICIES = len(policies_name)
cumSumRwd = zeros((N_POLICIES, N_REPETITIONS, HORIZON)) 

for i,p in enumerate(policies):
    c_print("Run {}/{}. Policy: {}".format(i,len(policies_name), policies_name[i]))
    evaluation = Evaluation(mab, p, HORIZON, N_REPETITIONS)
    

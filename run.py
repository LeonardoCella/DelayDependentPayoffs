__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.tools.io import c_print
from adaptiveRank.arm import Bernoulli
from adaptiveRank.Evaluation import Evaluation
from adaptiveRank.environment import MAB
from adaptiveRank.policies.Random import Random
from adaptiveRank.policies.Greedy import Greedy

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
parser.add_option('-T', dest = 'T', default = 100, type = "int", help = "Time horizon")
parser.add_option('-k', dest = 'N_BUCKETS', default = 10, type = "int", help = "Number of buckets")
parser.add_option('--nrep', dest = 'NREP', default = 1, type = "int", help = "Number of repetitions")
(opts, args) = parser.parse_args()

GAMMA = opts.GAMMA
MAX_DELAY = opts.MAX_DELAY
HORIZON = opts.T
N_BUCKETS = opts.N_BUCKETS
N_REPETITIONS = opts.NREP
N_POLICY_THREAD = 1
assert N_BUCKETS >= MAX_DELAY, "The number of arms cannot be lower than the max delay"
c_print(2, "run horizon: {}, number of buckets: {}, max delay: {}, gamma {}".format(HORIZON, N_BUCKETS, MAX_DELAY, GAMMA))

#=====================
# INITIALIZATION 
#====================
mab = MAB(HORIZON, N_BUCKETS, N_REPETITIONS, GAMMA, MAX_DELAY)
policies = [Random(), Greedy(), Greedy()]
policies_name = ['Random', 'Greedy', 'Ghost']
assert len(policies) == len(policies_name), "Check consistency of policy naming"
N_POLICIES = len(policies_name)
cumSumRwd = zeros((N_POLICIES, N_REPETITIONS, HORIZON)) 
exit()
for i,p in enumerate(policies):
    c_print(2, "Run {}/{}. Policy: {}".format(i,len(policies_name), policies_name[i]))
    evaluation = Evaluation(mab, p, HORIZON, N_REPETITIONS, N_POLICY_THREAD)

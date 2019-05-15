__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.tools.io import c_print
from adaptiveRank.arm import Bernoulli
from adaptiveRank.Evaluation import Evaluation
from adaptiveRank.environment import MAB
from adaptiveRank.policies.UCB import UCB
from adaptiveRank.policies.Ghost import Ghost
from adaptiveRank.policies.Greedy import Greedy
from adaptiveRank.policies.FPO_UCB import FPO_UCB

from optparse import OptionParser
from numpy import mean, std, zeros, arange
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib

#====================
# RUNNING PARAMETERS
#====================
parser = OptionParser(usage="usage: %prog [options]",
        version="%prog 1.0")
parser.add_option('--gamma', dest = 'GAMMA', default = 0.5, type = "float", help = "Discount parameter")
parser.add_option('--max_delay', dest = 'MAX_DELAY', default = 5, type = "int", help = "Memory size for the discount")
parser.add_option('--tau', dest = 'TAU', default = '4', type = 'int', help = 'Sampling delay')
parser.add_option('-T', dest = 'T', default = 100, type = "int", help = "Time horizon")
parser.add_option('-k', dest = 'N_BUCKETS', default = 10, type = "int", help = "Number of buckets")
parser.add_option('--fra_top', dest = 'FRA_TOP', default = 0.4, type = "float", help = "Fraction of top arms")
parser.add_option('--delay_ub', dest = 'DELAY_UB', default = 2, type = "int", help = "Gap from the delay bar")
parser.add_option('--delta', dest = "DELTA", default = 0.1, type = "float", help = "confidence in estimates")
parser.add_option('--n_rep', dest = 'N_REP', default = 1, type = "int", help = "Number of repetitions")
parser.add_option('--rounding', dest = 'ROUNDING', default = 5, type = "int", help = "Number of kept decimals")
parser.add_option('--app', dest = 'APPROXIMATE', default = 1, type = "int", help = "Approximation flag")
parser.add_option('-v', dest = 'VERBOSE', default = '5', type = 'int', help = "Verbose in terms of plots")
(opts, args) = parser.parse_args()

# Parsing parameters
GAMMA = opts.GAMMA
MAX_DELAY = opts.MAX_DELAY
DELTA = opts.DELTA
TAU = opts.TAU
FRA_TOP = opts.FRA_TOP
DELAY_UB = opts.DELAY_UB + MAX_DELAY
HORIZON = opts.T
N_BUCKETS = opts.N_BUCKETS
N_REPETITIONS = opts.N_REP
ROUNDING = opts.ROUNDING
VERBOSE = opts.VERBOSE
APPROXIMATE = opts.APPROXIMATE
assert TAU <= DELAY_UB, "Tau must not be greater than the d_bar upper bound"

#=====================
# INITIALIZATION 
#===================== 
policies = [Greedy(2), Ghost(2), UCB(2)] 
policies_name = ['Greedy', 'Ghost', 'UCB1']
#policies = []
#policies_name = []

policies.append(FPO_UCB(TAU, DELAY_UB, DELTA, ROUNDING, 5, APPROXIMATE))
policies_name.append('FPO')
assert len(policies) == len(policies_name), "Check consistency of policy naming"
N_POLICIES = len(policies_name)
cumSumRwd = zeros((N_POLICIES, N_REPETITIONS, HORIZON))

#=====================
# RUN OVER POLICIES
#=====================
results = []
for i,p in enumerate(policies):
    mab = MAB(HORIZON, N_BUCKETS, GAMMA, FRA_TOP, MAX_DELAY, APPROXIMATE)
    c_print(5, "=========RUN_POLICIES=========")
    c_print(5, "===Run.py, Run {}/{}. Policy: {}".format(i,len(policies_name)-1, policies_name[i]))
    results.append(Evaluation(mab, p, HORIZON, policies_name[i], N_REPETITIONS).getResults())

#=====================
# PLOTTING RESULTS
#=====================
if opts.VERBOSE:
    COLORS = ['b', 'g', 'r', 'y']
    MARKERS = ['o', '+', 'x', 'v']
    plt_fn =  plt.plot
    fig = plt.figure()
    plt.title("Arms: {}, Gamma: {}, Max Delay: {}, Fraction Top-Arms: {}".format(mab.nbArms, GAMMA, MAX_DELAY, FRA_TOP))
    ax = fig.add_subplot(1,1,1)
    i = 0
    for name,avg,std in results:
        plt.fill_between(arange(HORIZON), avg - (std/2), avg + (std/2), alpha = 0.5, color = COLORS[i])
        plt_fn(arange(HORIZON), avg, color = COLORS[i], marker = MARKERS[i], markevery=HORIZON/10, label = name)
        i+=1
    plt.legend(loc=2)
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Reward')
    plt.grid()
    plt.show()

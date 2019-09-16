__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.tools.io import c_print
from adaptiveRank.arm import Bernoulli
from adaptiveRank.Evaluation import Evaluation
from adaptiveRank.environment import MAB
from adaptiveRank.policies.UCB import UCB
from adaptiveRank.policies.Ghost import Ghost
from adaptiveRank.policies.RStar import RStar
from adaptiveRank.policies.Greedy import Greedy
from adaptiveRank.policies.FPO_UCB import FPO_UCB
from adaptiveRank.policies.Ore import ORE2

from optparse import OptionParser
from numpy import mean, std, zeros, arange, where
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib

#====================
# RUNNING PARAMETERS
#====================
parser = OptionParser(usage="usage: %prog [options]",
        version="%prog 1.0")
parser.add_option('--gamma', dest = 'GAMMA', default = 0.999, type = "float", help = "Discount parameter")
parser.add_option('--max_delay', dest = 'MAX_DELAY', default = 6, type = "int", help = "Memory size for the discount")
parser.add_option('--tau', dest = 'TAU', default = '7', type = 'int', help = 'Sampling delay')
parser.add_option('-T', dest = 'T', default = 500000, type = "int", help = "Time horizon")
parser.add_option('-k', dest = 'N_BUCKETS', default = 8, type = "int", help = "Number of buckets")
parser.add_option('--fra_top', dest = 'FRA_TOP', default = 0.2, type = "float", help = "Fraction of top arms")
#parser.add_option('--delay_ub', dest = 'DELAY_UB', default = 2, type = "int", help = "Gap from the delay bar")
parser.add_option('--delta', dest = "DELTA", default = 0.1, type = "float", help = "confidence in estimates")
parser.add_option('--n_rep', dest = 'N_REP', default = 1, type = "int", help = "Number of repetitions")
parser.add_option('--rounding', dest = 'ROUNDING', default = 5, type = "int", help = "Number of kept decimals")
parser.add_option('--bin', dest = 'BINARY', default = 1, type = "int", help = "Binary rewards")
parser.add_option('-v', dest = 'VERBOSE', default = '1', type = 'int', help = "Verbose in terms of plots")
parser.add_option('-s', dest = 'STORE', default = '1', type = 'int', help = "Storing plots")
parser.add_option('--stage', dest = 'MOD', default = '0', type = 'int', help = "0 - full learning, 1 arm ordering, 2 rank estimation")
parser.add_option('--test', dest = 'TEST', default = '0', type = 'int', help = "testing with specified means")
(opts, args) = parser.parse_args()

# Parsing parameters
GAMMA = opts.GAMMA
MAX_DELAY = opts.MAX_DELAY
DELTA = opts.DELTA
TAU = opts.TAU
FRA_TOP = opts.FRA_TOP
HORIZON = opts.T
N_BUCKETS = opts.N_BUCKETS
N_REPETITIONS = opts.N_REP
ROUNDING = opts.ROUNDING # number of kept decimals
VERBOSE = opts.VERBOSE
STORE = opts.STORE
BINARY = opts.BINARY # Binary rewards
MOD = opts.MOD # Running modality
TEST = opts.TEST # Given arms

#=====================
# INITIALIZATION 
#===================== 
if MOD != 1: # Useless benchmarks for the arm ordering estimation problem
    policies = [RStar(HORIZON, 2)]#, UCB(HORIZON, 2)]
    policies_name = ['Ghost']#, 'UCB1']
else:
    policies = []
    policies_name = []

# Appending an additional benchmark
policies.append(FPO_UCB(HORIZON, TAU, DELTA, ROUNDING, 5, BINARY, MOD, 10))
policies_name.append('PI ucb')
policies.append(ORE2(HORIZON, TAU, DELTA, 50, ROUNDING, 5, BINARY, MOD))
policies_name.append('PI Low')

assert len(policies) == len(policies_name), "Check consistency of policy naming"
N_POLICIES = len(policies_name)
cumSumRwd = zeros((N_POLICIES, N_REPETITIONS, HORIZON))

#=====================
# RUN OVER POLICIES
#=====================
results = []
for i,p in enumerate(policies):
    mab = MAB(HORIZON, N_BUCKETS, GAMMA, FRA_TOP, MAX_DELAY, BINARY, MOD, policies_name[i], TEST)
    c_print(5, "=========RUN_POLICIES=========")
    c_print(5, "===Run.py, Run {}/{}. Policy: {}".format(i,len(policies_name)-1, policies_name[i]))
    evaluation = Evaluation(mab, p, HORIZON, policies_name[i], N_REPETITIONS)
    results.append(evaluation.getResults())
    nbArms = evaluation.getNbArms()

#=====================
# PLOTTING RESULTS
#=====================
if opts.VERBOSE:
    COLORS = ['b', 'g', 'r', 'y', 'k', 'c', 'm']
    MARKERS = ['o', '+', 'x', 'v', 'o', '+', 'x']
    POLICY_AVGS = []
    POLICY_STD = []
    POLICY_N = []
    plt_fn =  plt.plot
    fig = plt.figure(1)
    plt.title("Gamma: {}, Max Delay: {}, Arms: {}, Fraction Top-Arms: {}".format(GAMMA, MAX_DELAY, nbArms, FRA_TOP))
    ax = fig.add_subplot(1,1,1)
    i = 0
    for name,avg,std in results:
        POLICY_AVGS.append(avg)
        POLICY_STD.append(std)
        POLICY_N.append(name)
        plt.fill_between(arange(HORIZON), avg - (std/2), avg + (std/2), alpha = 0.5, color = COLORS[i])
        plt_fn(arange(HORIZON), avg, color = COLORS[i], marker = MARKERS[i], markevery=HORIZON/100, label = name)
        i+=1
    plt.legend(loc=2)
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Reward')
    plt.grid()
    if STORE > 0:
        prefix = ['full', 'arm_ordering', 'rank_estimation']
        plt.savefig("output/{}_g{}_ft{}_d{}_dUB{}_dBar{}_bin{}_T{}_k{}.png".format(prefix[MOD], GAMMA, FRA_TOP, DELTA, MAX_DELAY, TAU, BINARY, HORIZON, N_BUCKETS))

        if MOD != 1: # Not Arm Ordering
            ghost_index = POLICY_N.index('Ghost')
            piucb_index = POLICY_N.index('PI ucb')
            pilow_index = POLICY_N.index('PI Low')

            avg_regret_ucb = POLICY_AVGS[ghost_index] - POLICY_AVGS[piucb_index]
            avg_regret_low = POLICY_AVGS[ghost_index] - POLICY_AVGS[pilow_index]

            std_regret_ucb = POLICY_STD[ghost_index] + POLICY_STD[piucb_index]
            std_regret_low = POLICY_STD[ghost_index] + POLICY_STD[pilow_index]


            # Regret figure creation
            plt_fn = plt.plot
            fig = plt.figure()
            plt.title("Regret wrt Ghost, Gamma {}, Arms {}, Fra. Top-Arms {}".format(GAMMA, MAX_DELAY, nbArms, FRA_TOP))
            ax = fig.add_subplot(1,1,1)

            # PI UCB plot
            plt.fill_between(arange(HORIZON), avg_regret_ucb - (std_regret_ucb/2), avg_regret_ucb + (std_regret_ucb/2), alpha = 0.5, color = COLORS[piucb_index])
            plt_fn(arange(HORIZON), avg_regret_ucb, color = COLORS[piucb_index], marker = MARKERS[piucb_index], markevery=HORIZON/100, label = 'Pi ucb')

            # PI low plot
            plt.fill_between(arange(HORIZON), avg_regret_low - (std_regret_low/2), avg_regret_low + (std_regret_low/2), alpha = 0.5, color = COLORS[pilow_index])
            plt_fn(arange(HORIZON), avg_regret_low, color = COLORS[pilow_index], marker = MARKERS[pilow_index], markevery=HORIZON/100, label = 'Pi Low')

            plt.legend(loc=2)
            plt.xlabel('Rounds')
            plt.ylabel('Regret')
            plt.grid()
            ax.set_xscale('log')
            plt.savefig("output/regret_g{}_ft{}_d{}_dUB{}_dBar{}_bin{}_T{}_k{}.png".format(GAMMA, FRA_TOP, DELTA, MAX_DELAY, TAU, BINARY, HORIZON, N_BUCKETS))

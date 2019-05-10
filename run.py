__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.tools.io import c_print
from adaptiveRank.arm import Bernoulli
from adaptiveRank.Evaluation import Evaluation
from adaptiveRank.environment import MAB
from adaptiveRank.policies.UCB import UCB
from adaptiveRank.policies.Ghost import Ghost
from adaptiveRank.policies.Greedy import Greedy

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
parser.add_option('-T', dest = 'T', default = 100, type = "int", help = "Time horizon")
parser.add_option('-k', dest = 'N_BUCKETS', default = 10, type = "int", help = "Number of buckets")
parser.add_option('--fratop', dest = 'FRA_TOP', default = 0.4,type = "float", help = "Fraction of top arms")
parser.add_option('--nrep', dest = 'NREP', default = 1, type = "int", help = "Number of repetitions")
(opts, args) = parser.parse_args()

GAMMA = opts.GAMMA
MAX_DELAY = opts.MAX_DELAY
FRA_TOP = opts.FRA_TOP
HORIZON = opts.T
N_BUCKETS = opts.N_BUCKETS
N_REPETITIONS = opts.NREP
assert N_BUCKETS >= MAX_DELAY, "The number of arms cannot be lower than the max delay"
c_print(2, "Run.py, Parameters: horizon: {}, number of buckets: {}, max delay: {}, gamma {}\n".format(HORIZON, N_BUCKETS, MAX_DELAY, GAMMA))

#=====================
# INITIALIZATION 
#=====================
policies = [UCB(), Greedy(), Ghost()]
policies_name = ['UCB1', 'Greedy', 'Ghost']
assert len(policies) == len(policies_name), "Check consistency of policy naming"
N_POLICIES = len(policies_name)
cumSumRwd = zeros((N_POLICIES, N_REPETITIONS, HORIZON)) 

#=====================
# RUN OVER POLICIES
#=====================
results = []
for i,p in enumerate(policies):
    mab = MAB(HORIZON, N_BUCKETS, GAMMA, FRA_TOP, MAX_DELAY)
    c_print(5, "=========RUN_POLICIES=========")
    c_print(5, "===Run.py, Run {}/{}. Policy: {}".format(i,len(policies_name)-1, policies_name[i]))
    results.append(Evaluation(mab, p, HORIZON, policies_name[i], N_REPETITIONS).getResults())

#=====================
# PLOTTING RESULTS
#=====================
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

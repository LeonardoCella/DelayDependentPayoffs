''' Environment for a multi-armed bandit problem'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.environment.Environment import Environment
from adaptiveRank.Results import *
from adaptiveRank.tools.io import c_print
from adaptiveRank.arm.Bernoulli import Bernoulli

from numpy import linspace, around, array

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, horizon, nbarms, nRepetitions, gamma, max_delay):
        self.horizon = horizon
        self.nbArms =  nbarms
        self.gamma = gamma
        self.max_delay = max_delay
        self._nbRepetitions = nRepetitions

        # Arm Creation
        means = linspace(0.0, 1.0, self.nbArms, endpoint = False)
        c_print(2, "Arm default means: {}".format(means))
        arms = []
        means = around(array(means), 3)
        for mu in means:
            tmp_arm = Bernoulli(mu, self.gamma, self.max_delay)
            arms.append(tmp_arm)
            c_print(2, "MAB: Created arm: {}".format(tmp_arm))
        c_print(1, "List of {} Bernoulli arms".format(len(arms)))


    def play(self, policy, horizon, nbRepetition):
        ''' Called once per policy from __init__ of Evaluation. Is the one that manages the rounds.'''


        return result

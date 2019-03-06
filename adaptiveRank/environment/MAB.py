''' Environment for a multi-armed bandit problem'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.environment.Environment import Environment
from adaptiveRank.Results import *
from adaptiveRank.tools.io import c_print
from adaptiveRank.arm.Bernoulli import Bernoulli

from numpy import linspace, around, array, zeros

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, horizon, nbarms, nRepetitions, gamma, max_delay):
        self.horizon = horizon
        self.nbArms =  nbarms
        self.gamma = gamma
        self.maxDelay = max_delay
        self._nbRepetitions = nRepetitions
        self._armsDelay = zeros(nbarms)
        self._armsStates = zeros(nbarms)

        # Arm Creation
        means = linspace(0.0, 1.0, self.nbArms, endpoint = False)
        c_print(2, "Arm default means: {}".format(means))
        self.arms = []
        means = around(array(means), 3)
        for mu in means:
            tmpArm = Bernoulli(mu, self.gamma, self.maxDelay)
            self.arms.append(tmpArm)
            c_print(2, "MAB: Created arm: {}".format(tmpArm))
        c_print(1, "List of {} Bernoulli arms".format(len(arms)))

    def _compute_states(self):
        for arm in self.arms:
            self._currentArmValues = [arm.draw() for arm in self.arms]

    def 

    def play(self, policy, horizon, nbRepetition):
        ''' Called once per policy from __init__ of Evaluation. Is the one that manages the rounds.'''
        for t in range(horizon):
            states = self.compute_states()

        return result

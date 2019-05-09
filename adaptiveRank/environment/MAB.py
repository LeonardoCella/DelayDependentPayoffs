''' Environment for a multi-armed bandit problem'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.environment.Environment import Environment
from adaptiveRank.Results import *
from adaptiveRank.tools.io import c_print
from adaptiveRank.arm.Bernoulli import Bernoulli

from numpy import arange, around, array, linspace, unique, zeros
import sortednp as snp

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, horizon, nbBuckets, nRepetitions, gamma, maxDelay):
        self.horizon = horizon
        self.nbBuckets = nbBuckets
        self.gamma = gamma
        self.maxDelay = maxDelay
        self._nbRepetitions = nRepetitions
        self.nbArms = self._arm_creation()
        self._armsDelay = zeros(self.nbArms)
        self._armsIndexes = arange(self.nbArms)
        c_print(2, "MAB INIT: List of {} Bernoulli arms".format(len(self.arms)))
        self._armsStates = zeros(self.nbArms)

    def _arm_creation(self):
        # Arm Creation
        full_grid = linspace(0.0, 1.0, self.nbBuckets, endpoint = True)
        c_print(1, "Buckets: {}".format(full_grid))
        if self.maxDelay < self.nbBuckets:
            delta = 1.0/(self.nbBuckets)# Previously adopted delta
            new_extreme = delta*self.maxDelay
            good_arms = linspace(0.0, new_extreme, self.nbBuckets, endpoint = False)
            c_print(1, "Good arms: {}".format(good_arms))
            means = around(array(snp.merge(full_grid, good_arms)), 2) # evenly round to 2 decimals 
        else:
            means = full_grid 
        self.arms = []
        means = unique(means)
        c_print(2, "Arm means: {}".format(means))
        for mu in means:
            tmpArm = Bernoulli(mu, self.gamma, self.maxDelay)
            self.arms.append(tmpArm)
            c_print(1, "MAB INIT: Created arm: {}".format(tmpArm))
        return len(means)

    def compute_states(self):
        '''Called at every step in the play() method. Manages the trajectory evolution.'''
        assert len(self._armsDelay) == len(self._armsStates), "MAB compute_states: Incoherent size"
        for i, arm, delay in zip(arange(0, self.nbBuckets, 1), self.arms, self._armsDelay):
            print(i, arm, delay)
            self._armsStates[i] = arm.computeState(delay)
        c_print(2, "MAB compute_state() Arms states: {}".format(self._armsStates))

    def play(self, policy, horizon, nbRepetition):
        ''' Called once per policy from __init__ of Evaluation. Is the one that manages the rounds.'''
        c_print(2, "MAB play()")

        result = Result(horizon)
        for t in range(horizon):
            self.compute_states()
            c_print(2, "ArmsStates type {} shape {}".format(type(self._armsStates), self._armsStates.shape))

            # Choice and Feedback 
            choice = policy.choice(self._armsStates)
            reward = self.arms[choice].draw(self._armsDelay[choice])
            result.store(t, choice, reward)

            # Delays update
            for i in self._armsIndexes:
                d = self._armsDelay[i]
                if d != 0 and i != choice:
                    self._armsDelay[i] += 1
                if d > self.maxDelay:
                    self._armsDelay[i] = 0
                if i == choice:
                    self._armsDelay[i] = 1

        return result

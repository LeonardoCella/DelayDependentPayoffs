''' Environment for a multi-armed bandit problem'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.environment.Environment import Environment
from adaptiveRank.Results import *
from adaptiveRank.tools.io import c_print
from adaptiveRank.arm.Bernoulli import Bernoulli

from numpy import around, array, arange, linspace, zeros

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, horizon, nbArms, nRepetitions, gamma, maxDelay):
        self.horizon = horizon
        self.nbArms =  nbArms
        self.gamma = gamma
        self.maxDelay = maxDelay
        self._nbRepetitions = nRepetitions
        self._armsDelay = zeros(nbArms)
        self._armsIndexes = arange(nbArms)
        self._armsStates = zeros(nbArms)

        # Arm Creation
        means = linspace(0.0, 1.0, self.nbArms, endpoint = False)
        c_print(2, "Arm default means: {}".format(means))
        c_print(2, "Arm default indexes: {}".format(self._armsIndexes))
        self.arms = []
        means = around(array(means), 3)
        for mu in means:
            tmpArm = Bernoulli(mu, self.gamma, self.maxDelay)
            self.arms.append(tmpArm)
            c_print(2, "MAB INIT: Created arm: {}".format(tmpArm))
        c_print(1, "MAB INIT: List of {} Bernoulli arms".format(len(self.arms)))

    def compute_states(self):
        '''Called at every step in the play() method. Manages the trajectory evolution.'''
        assert len(self._armsDelay) == len(self._armsStates), "MAB compute_states: Incoherent size"
        for i, arm, delay in zip(arange(0, self.nbArms, 1), self.arms, self._armsDelay):
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
                if i == choice:
                    self._armsDelay[i] = 1

        return result

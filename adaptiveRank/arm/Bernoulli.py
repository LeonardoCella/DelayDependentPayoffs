'''Bernoulli distributed arm.'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

from scipy.stats import bernoulli

from adaptiveRank.tools.io import c_print
from adaptiveRank.arm.Arm import Arm

TEST = True

class Bernoulli(Arm):
    """Bernoulli distributed arm"""

    def __init__(self, mean, gamma, maxDelay):
        self._mean = mean
        self._gamma = gamma
        self._maxDelay = maxDelay

    def __str__(self):
        return "Bernoulli arm. mu: {} gamma: {} max_delay: {}".format(self._mean, self._gamma, self._maxDelay)

    def draw(self, currentDelay):
        expectedReward = self._mean
        if currentDelay != 0 and currentDelay <= self._maxDelay:
            c_print(1, "Discounting")
            expectedReward *= (1 - self._gamma**currentDelay)
        return expectedReward
        #return bernoulli.rvs(expectedReward)

    def computeState(self, currentDelay):
        expectedReward = self._mean
        if currentDelay != 0 and currentDelay <= self._maxDelay:
            c_print(1, "Discounting")
            expectedReward *= (1 - (self._gamma**currentDelay))
        return expectedReward

'''Bernoulli distributed arm.'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

from scipy.stats import bernoulli
from adaptiveRank.arm.Arm import Arm

TEST = True

class Bernoulli(Arm):
    """Bernoulli distributed arm"""

    def __init__(self, mean, gamma, maxdelay):
        self._mean = mean
        self._gamma = gamma
        self._maxdelay = maxdelay

    def draw(self, current_delay):
        draw = bernoulli.rvs(mean, size = 1)
        if current_delay <= maxdelay:
            draw *= (1 - self._gamma**current_delay)
        return draw

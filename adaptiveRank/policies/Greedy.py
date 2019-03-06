''' First random policy'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

import numpy as np

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class Greedy(Policy):
    def __init__(self):
        pass

    def choice(self, arms):
        return arms.argmax()

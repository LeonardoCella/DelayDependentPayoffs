''' First random policy'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

import numpy as np
import random

from adaptiveRank.polcies.Policy import Policy
from adaptiveRank.tools.io import c_print

class Random(Policy):
    def __init__(self):
        pass

    def choice(self, arms):
        return arms.index(random.choice(arms))

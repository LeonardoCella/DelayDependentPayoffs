''' Ghost policy'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

import numpy as np

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class Ghost(Policy):
    def __init__(self, MOD = 1):
        self.cIndex = -1 # last round robin index
        self.r = -1 # rank
        self.MOD = MOD # Specifies the running mod

    def choice(self, arms):
        # First round robin condition
        if self.r == -1:
            index = arms.argmax()
            # RR termination condition
            if index <= self.cIndex:
                c_print(4, "Ghost rank: {}".format(self.cIndex))
                self.r = self.cIndex
        else:
            index = self.cIndex + 1
            if index > self.r:
                index = 0

        self.cIndex = index
        return [index]

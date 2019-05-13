''' FastPartialOrder plus max-rank policy'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

import numpy as np
from math import ceil

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class FPO_UCB(Policy):
    '''FastPartialOrder and MaxRank'''

    def __init__(self, tau, delay_ub, MOD):
        # Non-stationarity parameters
        self.delay_ub = delay_ub
        self.tau = tau

        # Running parameters
        self.t = 0
        self.delta = 0.1
        self.MOD = MOD

        # Policy "state"
        self.filtered_arms = [] # arm indexes

    def _bucketing(self, indexes):
        returned_list = []
        for i in range(ceil(len(indexes)/self.tau)): # number of chunks
            l = min((len(indexes)-1, (i+1)*self.tau))
            returned_list += indexes[i*self.tau : l]
            returned_list += indexes[i*self.tau : l]
        print(returned_list)
        return returned_list

    def choice(self, arms):
        if self.t == 0: # Initialization
            self.filtered_arms = np.arange(0, len(arms), 1)
            c_print(1, "Array {}, len {}".format(self.filtered_arms, len(arms)))
        self.t = self.t + 1

        # Stage scheduler
        if len(self.filtered_arms) > self.delay_ub: # Fast Partial Order
            index = self._fpo(arms)
        else: # Max Rank
            index = self._maxrank(arms)

        return self._bucketing(index)

    def update(self, arm, rwd, delay):
        # Unbiased updates
        if delay == self.tau:
            c_print(1, "Storing {} for {} with delay {}".format(rwd, arm, delay))

    def _fpo(self, arms):
        return [1,2,3,4,5,6,7]

    def _maxrank(self, arms):
        return 1

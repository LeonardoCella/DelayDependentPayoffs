''' FastPartialOrder plus max-rank policy'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

from math import ceil, log, sqrt
from random import choice
from numpy import arange, argpartition, argmax, argsort, array, ones, where, zeros

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class FPO_UCB(Policy):
    '''FastPartialOrder and MaxRank'''

    def __init__(self, tau, delayUb, delta = 0.1, rounding = 5, MOD = 2, approximate = False):
        c_print(4, "\nFPO_UCB Init. Tau {}, DelayUB {}, delta {}".format(tau, delayUb, delta))
        # Non-stationarity parameters
        self.delayUb = delayUb
        self.tau = tau

        # Running parameters
        self.t = 0 # round iterator
        self.s = 0 # phase iterator
        self._delta = delta
        self._rounding = rounding
        self._MOD = MOD
        self._APP = approximate

        # Policy "state"
        self._activeArms = [] # mean ascending sorted arm indexes
        self.ranks = [i+1 for i in range(self.delayUb)]
        assert len(self.ranks) == self.delayUb, "wrong rank set construction"
        self._nbPullsRanks = [0] * self.delayUb
        self._learnedPO = False

    def choice(self, arms):
        if self.t == 0: # Data Structures initialization
            self._nArms = len(arms)
            self._activeArms = [i for i in range(self._nArms)]
            self._nbPullsArms = [0] * self._nArms
            self._cumRwdArms = [0.0] * self._nArms
            self._meanArms = [0.0] * self._nArms
            self._cumRwdArmDelay = zeros((self._nArms, self.delayUb + 1))
            self._nbPullsArmDelay = zeros((self._nArms, self.delayUb + 1))
            idx = self._bucketing(self._activeArms)
            # Each arm is played once 
            c_print(self._MOD, "FPO.py, choice(): First Pull, round {}, pulling {}".format(self.t, idx))
            return idx

        # Eventual Free elimination
        if len(self._activeArms) > self.delayUb:
            c_print(1, "FPO.py, choice(): Calling Discarding(), active arms {}, delayUB {}".format(self._activeArms, self.delayUb))
            self._discarded()

        if len(self._activeArms) - self.s > self.delayUb:
            idx = self._bucketing(self._activeArms)
            c_print(1, "FPO.py, choice(): Best {}, round {}, pulls: {}".format(self.delayUb, self.t, idx))
            return idx
        else: # 2nd stage FPO or Max-Rank 
            if not self._ordered():
                index = self._activeArms
                idx = self._bucketing(index)
                c_print(1, "FPO.py, choice(): PO, round {} cb {}, pulls: {}".format(self.t, self._cb(), idx))
                return idx
            else:
                self._learnedPO = True
                index = list(self._maxrank())
                index.extend(index)
                c_print(1, "FPO.py, choice(): round {} Max_Rank {}".format(self.t, index))
                return index

    def _discarded(self):
        assert self._nArms - self.s == len(self._activeArms), "Inconsistent arm elimination"
        sorted_idx = argsort(self._meanArms)[::-1] #Descending empirical mean order 
        # Gap computation
        gap = self._meanArms[sorted_idx[self.delayUb-1]] - self._meanArms[sorted_idx[self._nArms-self.s-1]]
        gap = round(gap, self._rounding)

        if gap > self._cb():  # Discarding index 
            c_print(4, "FPO.py, discarding(): Gap {} arms: {}-{} with cb {} activeArms {} sorted_idx {} means {}".format(gap, sorted_idx[self.delayUb-1], sorted_idx[self._nArms-self.s-1], self._cb(), self._activeArms, sorted_idx, self._meanArms))
            # Removing the index from the active arms
            idx = self._activeArms.remove(sorted_idx[self._nArms - self.s -1])
            self.s = self.s + 1
            if len(self._activeArms) == self.delayUb:
                c_print(4, "FPO.py, discarding() Found Best {} Arms. Active Arms: {}, Phase {}".format(self.delayUb, self._activeArms, self.s))
        return

    def _ordered(self):
        sorted_idx = argsort(self._meanArms)[::-1]
        for i in range(len(self._activeArms) - 1):
            gap = self._meanArms[sorted_idx[i]] - self._meanArms[sorted_idx[i+1]]
            gap = round(gap, self._rounding)
            if gap < self._cb():
                c_print(1, "FPO.py, _ordered(): Gap between {} and {} is {} with cb {}".format(self._activeArms[i], self._activeArms[i-1], gap, self._cb()))
                return False
        if not self._learnedPO:
            c_print(4, "FPO.py, _ordered(): Learned partial order. Active arms {}, sorted {}".format(self._activeArms, sorted_idx))
            self._activeArms = sorted_idx[:len(self._activeArms)]

        return True

    def _maxrank(self):
        pulls = array(self._nbPullsRanks)
        zero_idx = where(pulls == 0)[0]
        if len(zero_idx) > 0:
            index = choice(zero_idx)
            c_print(4, "FPO.py, _maxrank(): unpulled rank: {}".format(index+1))
            self._nbPullsRanks[index] += 1
        else:
            ucb_values = [0] * self.delayUb
            for rank in range(self.delayUb):
                mean_rank = 0.0
                for i in range(rank):
                    tmp = self._cumRwdArmDelay[self._activeArms[i],rank] / self._nbPullsArmDelay[self._activeArms[i],rank]
                    mean_rank = mean_rank + tmp
                ucb_values[rank] = mean_rank /(1+rank) + sqrt((2*log(self.t))/self._nbPullsRanks[rank])
            index = argmax(ucb_values)
            c_print(1, "FPO.py, _maxrank(): ucbs {} rank: {}".format(ucb_values, index+1))
        c_print(1, "FPO.py, _maxrank(): Pulling rank {} arms {}".format(index+1, self._activeArms[:index+1]))
        return self._activeArms[:index+1]

    def update(self, arm, rwd, delay):
        c_print(1, "FPO.py, update(): arm {} rwd {} delay {}".format(arm, rwd, delay))
        if not self._learnedPO: 
            # Unbiased updates
            if delay == self.tau:
                self.t = self.t + 1
                c_print(1, "FPO.py, update(): unbiased sample")
                self._cumRwdArms[arm] += rwd
                self._nbPullsArms[arm] = self._nbPullsArms[arm] + 1
                self._meanArms[arm] = self._cumRwdArms[arm]/self._nbPullsArms[arm]
        else: # Max Rank stage
            self.t = self.t + 1
            c_print(1, "FPO.py, update(): rank rwd {} arm {} with delay {}".format(rwd, arm, delay))
            if delay > self.delayUb:
                delay = 1
            self._cumRwdArmDelay[arm,int(delay)-1] += rwd
            self._nbPullsArmDelay[arm, int(delay)-1] += 1

    def _bucketing(self, indexes):
        returned_list = []
        len_idx = len(indexes)
        n_full_chunks = int(len_idx/self.tau)
        bucketed = []

        if len(indexes) != self.tau:
            c_print(1, "Number of full chunks {}".format(n_full_chunks))
            for i in range(n_full_chunks):
                c_print(1, "Inserting in full chunks")
                l = min((len_idx, (i+1)*self.tau))
                returned_list += indexes[i*self.tau : l]
                returned_list += indexes[i*self.tau : l]

        # Fill the eventual last partial piece of the list with not pulled active arms
        if ceil(len(indexes)/self.tau) != int(len_idx/self.tau) or self.tau == len(indexes):
            c_print(1, "FPO.py, bucketing(), Completing the last chunk")
            remaining_indexes = indexes[n_full_chunks*self.tau : min(len_idx, (n_full_chunks+1)*self.tau)]
            c_print(1, "FPO.py, bucketing(), Remaining indexes {}".format(remaining_indexes))
            i = 0
            bucketed = remaining_indexes
            while len(bucketed) < self.tau:
                if self._activeArms[i] not in remaining_indexes:
                    bucketed.append(self._activeArms[i])
                    c_print(1, "Index i {}, bucketed {}".format(self._activeArms[i], bucketed))
                i = i + 1
            returned_list += bucketed + bucketed
        c_print(1, "FPO.py, bucketing(), Bucketed List: {}".format(returned_list))

        assert len(returned_list) == 2*self.tau*ceil(len_idx/self.tau), "Inconsistent bucketing"
        return returned_list

    def _cb(self):
        return 2*round(sqrt((log(((len(self._activeArms))*(1+self.t))/(self._delta)))/(1+self.t)), self._rounding)

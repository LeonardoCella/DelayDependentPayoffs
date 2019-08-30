''' FastPart:ialOrder plus max-rank policy'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

from math import ceil, log, sqrt
from random import choice
from numpy import arange, argpartition, argmax, argsort, array, ones, where, zeros

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class ORE2(Policy):
    '''Ordering and Rank Estimation via Elimination'''

    def __init__(self, T, tau, delta = 0.1, rounding = 5, MOD = 2, approximate = False):
        c_print(4, "\nFPO_UCB Init. Tau {}, DelayUB {}, delta {}".format(tau, delayUb, delta))
        # Non-stationarity parameters
        self._tau = tau

        # Running parameters
        self.horizon = T # horizon
        self.delta = delta # confidence
        self.rounding = rounding # rounding approximation
        self.MOD = MOD
        self.APP = approximate

        # Policy "state"
        self._t = 0 # round iterator, necessary for filtering biased rewards
        self._r = 0 # stage iterator during rank elimination
        self._s = 0 # stage iterator during 1st phase of PO estimation
        # Policy "state": RANKS Data Structures
        self._activeArms = [] # mean ascending sorted arm indexes
        self._activeRanks = [] # rank indexes
        self._learnedPO = False # canary alerting the rank elimination stage


    def choice(self, arms):
        # First Round
        if self._t == 0: # Policy "state" Data Structures Construction
            self._nArms = len(arms)
            self._activeArms = [i for i in range(self._nArms)]
            self._activeRanks = [i for i in range(self._nArms)]
            self._nbPullsArms = [0] * self._nArms
            self._nbPullsRanks = [0] * self._nArms
            self._cumRwdArms = [0.0] * self._nArms
            self._cumRwdRanks = [0.0] * self._nArms
            self._meanArms = [0.0] * self._nArms
            self._meanRanks = [0.0] * self._nArms

            # Each arm is played once 
            idx = self._bucketing(self._activeArms)
            c_print(self.MOD, "PO.py, CHOICE(): First Pull, round {}, pulling {}".format(self._t, idx))
            self._r = self._r + 1
            assert self._r == 1, "Wrong sampling counter definition"
            return idx

        # Arm Elimination
        if self._samplingRequired():
            c_print(1, "PO.py, not ordered DISCARDING on active arms {}".format(self._activeArms))
            self._discarded()
            idx = self._bucketing(self._activeArms)
            self.r = self.r + 1
            return idx
        else: # Rank Elimination
            self._s = 1 # rank stages counter
            self._learnedPO = True
            index = list(self._maxrank()) # _maxrank() returns the list of top-x arms
            index.extend(index)
            c_print(4, "FPO.py, choice(): round {} Max_Rank {}\n".format(self._t, index))
            return index


    def _samplingRequired(self):
        sorted_idx = argsort(self._meanArms)[::-1] # descending sorted indexes
        for i in range(len(self._activeArms) - 1):
            gap = self._meanArms[sorted_idx[i]] - self._meanArms[sorted_idx[i+1]]
            gap = round(gap, self.rounding)
            if gap < self._cb(): # All gaps has to be greater than cb
                c_print(1, "FPO.py, ORDERED(): Gap between {} and {} is {} with cb {}".format(self._activeArms[i], self._activeArms[i-1], gap, self._cb()))
                return False

        # All arms satisfy the separation condition
        if not self._learnedPO:
            c_print(4, "FPO.py, ORDERED(): Learned partial order. Active arms {}, sorted {}".format(self._activeArms, sorted_idx))
            self._activeArms = sorted_idx[:len(self._activeArms)]

        return True


    def _discarded(self):
        assert self._nArms - self._s == len(self._activeArms), "Inconsistent arm elimination"
        sorted_idx = argsort(self._meanArms)[::-1] # all indexes sorted desc by means   
        len_activeArms = len(self._activeArms)
        arm_deletion = False # set Canary

        for i in range(len_activeArms - 1):

            if len(self._activeArms) == self._tau:
                idx = self._bucketing(self._activeArms)
                c_print(4, "FPO.py, DISCARDING(): Found Best {} Arms. Active Arms: {}, Phase {}".format(self.delayUb, self._activeArms, self._s))
                return
            if i != 0:
                gap_l = self._meanArms[sorted_idx[i-1]] - self._meanArms[sorted_idx[i]]
                gap_l = round(gap_l, self.rounding)
            if i != len_activeArms - 1:
                gap_r = self._meanArms[sorted_idx[i]] - self._meanArms[sorted_idx[i+1]]
                gap_r = round(gap_r, self.rounding)

            current_cb = self._cb()
            if i == 0:
                if gap_r > current_cb: # Condition for discarding the first index
                    arm_deletion = True
                    c_print(4, "FPO.py, DISCARDING(): Gap_Right {} vs CB {}, arm 0 index: {}, activeArms {} sorted_idx {} means {}".format(gap_r, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
            if i == len_activeArms - 1:
                if gap_l > current_cb: # Condition for discarding the last index
                    arm_deletion = True
                    c_print(4, "FPO.py, DISCARDING(): Gap_Left {} vs CB {}, last arm index: {}, activeArms {} sorted_idx {} means {}".format(gap_r, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
            if i!= len_activeArms - 1 and i!= 0:
                if gap_l > current_cb and gap_r > current_cb: # Conditino for discarding a in the middle index
                    arm_deletion = True
                    c_print(4, "FPO.py, DISCARDING(): Gap_Right {} Gap_Left {} vs CB {}, arm {} with Index: {}, activeArms {} sorted_idx {} means {}".format(gap_r, gap_l, current_cb, i, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))

            if arm_deletion: # Arm deletion based on one of previous cases
                idx = self._activeArms.remove(sorted_idx[i])
                self._s = self._s + 1


    def _maxrank(self):
        pulls = array(self._nbPullsRanks)
        zero_idx = where(pulls == 0)[0]
        if len(zero_idx) > 0:
            index = choice(zero_idx)
        else:
            ucb_values = [0.0] * self.delayUb
            # For each rank
            for rank in range(self.delayUb):
                mean_rank = 0.0
                # For the first rank-arms
                for i in self._activeArms[:(rank+1)]:
                    tmp = self._cumRwdArmDelay[i,rank] / self._nbPullsArmDelay[i,rank]
                    mean_rank = mean_rank + tmp
                ucb_values[rank] = mean_rank /(1+rank) + sqrt((2*log(self._t))/self._nbPullsRanks[rank])
            index = argmax(ucb_values)
        self._nbPullsRanks[index] += 1
        # Additional variables for updating with non-stationarities
        self._pulledRankIndex = index
        self._freezedTime = self._t
        c_print(4, "\nFPO.py, _maxrank(): Pulling rank {} arms {} rank_pulls {} ucb_values {}".format(index+1, self._activeArms[:index+1], self._nbPullsRanks, ucb_values))
        return self._activeArms[:index+1]

    def update(self, arm, rwd, delay):
        c_print(1, "FPO.py, update(): arm {} rwd {} delay {}".format(arm, rwd, delay))
        if not self._learnedPO:
            # Unbiased updates
            if delay == self._tau:
                self._t = self._t + 1
                c_print(1, "FPO.py, update(): unbiased sample")
                self._cumRwdArms[arm] += rwd
                self._nbPullsArms[arm] = self._nbPullsArms[arm] + 1
                self._meanArms[arm] = self._cumRwdArms[arm]/self._nbPullsArms[arm]
        else: # Max Rank stage
            self._t = self._t + 1
            time_gap = self._t - self._freezedTime
            # Windows of acceptance
            if time_gap  > self._pulledRankIndex + 1 and time_gap <= 2 * (self._pulledRankIndex + 1):
                c_print(1, "Storing Arm {} delay {}".format(arm, delay))
                self._cumRwdArmDelay[arm, self._pulledRankIndex] += rwd

                self._nbPullsArmDelay[arm, self._pulledRankIndex] += 1
        return

    def _bucketing(self, indexes):
        returned_list = []
        len_idx = len(indexes)
        n_full_chunks = int(len_idx/self._tau)
        bucketed = []

        if len(indexes) != self._tau:
            c_print(1, "Number of full chunks {}".format(n_full_chunks))
            # Bucketing the full chunks 
            for i in range(n_full_chunks):
                c_print(1, "Inserting in full chunks")
                l = min((len_idx, (i+1)*self._tau))
                returned_list += indexes[i*self._tau : l]
                returned_list += indexes[i*self._tau : l]

        # Fill the eventual last partial piece of the list with not pulled active arms
        if ceil(len(indexes)/self._tau) != int(len_idx/self._tau) or self._tau == len(indexes):
            c_print(1, "FPO.py, bucketing(), Completing the last chunk")
            remaining_indexes = indexes[n_full_chunks*self._tau : min(len_idx, (n_full_chunks+1)*self._tau)]
            c_print(1, "FPO.py, bucketing(), Remaining indexes {}".format(remaining_indexes))
            i = 0
            bucketed = remaining_indexes
            while len(bucketed) < self._tau:
                if self._activeArms[i] not in remaining_indexes:
                    bucketed.append(self._activeArms[i])
                    c_print(1, "Index i {}, bucketed {}".format(self._activeArms[i], bucketed))
                i = i + 1
            returned_list += bucketed + bucketed
        c_print(1, "FPO.py, bucketing(), Bucketed List: {}".format(returned_list))

        assert len(returned_list) == 2*self._tau*ceil(len_idx/self._tau), "Inconsistent bucketing"
        return returned_list

    def _cb(self):
        # Confidence bounds definition depending on the stage
        if self._learnedPO:
            return round(sqrt( log((self.delayUb * log(log(self.horizon)))/self.delta) * (self.delayUb / (2 * self._Ts()))), self.rounding)
        else:
            return round(sqrt(log((2 * self._nArms * self._r * (self._r+1))/(self.delta)) * (1/(2*self._r))), self.rounding)

    def _Ts(self):
        return self.horizon**(1 - 2**(-self._s))

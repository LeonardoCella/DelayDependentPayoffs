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

    def __init__(self, horizon, nbBuckets, gamma, fraTop, maxDelay, approximate):
        self.horizon = horizon
        self.nbBuckets = nbBuckets
        self.gamma = gamma
        self.fraTop = fraTop
        self.maxDelay = maxDelay
        self._approximate = approximate
        self.arms = []

        # Arm Creation
        self.nbArms = self._arm_creation()
        self._armsDelay = zeros(self.nbArms)
        self._armsIndexes = arange(self.nbArms)
        assert self._armsIndexes[-1] == self.nbArms -1, "Wrong arm creation"
        c_print(4, "MAB.py, INIT: Arm Indexes {}".format(self._armsIndexes))
        self._armsStates = zeros(self.nbArms)

    def _arm_creation(self):
        starting_grid = linspace(0.0, 1.0, self.nbBuckets, endpoint = True)
        c_print(1, "Buckets: {}".format(starting_grid))
        delta = 1.0/(self.nbBuckets) # Previously adopted delta
        new_extreme = delta*self.fraTop*self.nbBuckets
        good_arms = linspace(0.0, new_extreme, self.nbBuckets, endpoint = False)
        c_print(1, "Good arms: {}".format(good_arms))
        means = around(array(snp.merge(starting_grid, good_arms)), 2) # evenly round to 2 decimals 
        means = 1 - unique(means)
        c_print(4, "\n=========MAB_INIT=========")
        c_print(4, "MAB.py, Arm means: {}".format(means))
        for mu in means:
            tmpArm = Bernoulli(mu, self.gamma, self.maxDelay, self._approximate)
            self.arms.append(tmpArm)
            c_print(1, "MAB INIT: Created arm: {}".format(tmpArm))
        return len(means)


    def compute_states(self):
        '''Called at every step in the play() method. Manages the trajectory evolution.'''
        assert len(self._armsDelay) == len(self._armsStates), "MAB compute_states: Incoherent size"

        for i, arm, delay in zip(self._armsIndexes, self.arms, self._armsDelay):
            self._armsStates[i] = arm.computeState(delay)
            c_print(1, "MAB.py, states() Index {}, arm {}, delay {}".format(i, arm, delay))

        c_print(1, "MAB.py, states() {}".format(self._armsStates))
        return


    def play(self, policy, horizon, nbRepetition):
        ''' Called once per policy from __init__ of Evaluation. Rounds scheduler.'''
        c_print(1, "MAB.py, play()")

        result = Result(horizon)
        t = 0

        while t < horizon:
            c_print(1, "\n===\nMAB.py, play(): round {}\n===".format(t))

            if t == 0:
                self.compute_states()
                c_print(1, "MAB.py, play(): current delays: {}".format(self._armsDelay))
                c_print(1, "MAB.py, play(): arm states: {}".format(self._armsStates))

            # Structured Choice and Feedback 
            choice = policy.choice(self._armsStates)

            tmp = 0
            for c in choice:
                tmp = tmp + 1
                reward = self.arms[c].draw(self._armsDelay[c])

                c_print(1, "\nMAB.py, play(): Chosen arm: {} at round: {} with rwd {}".format(c, t, reward))
                c_print(1, "MAB.py play(), arm states: {}".format(self._armsStates))
                c_print(1, "MAB.py, play(): Suffered delays: {}".format(self._armsDelay))

                policy.update(c, reward, self._armsDelay[c])
                result.store(t, c, reward)

                # Delays update
                for i in self._armsIndexes:
                    d = self._armsDelay[i]
                    # Not the chosen arm and already pulled once
                    if d != 0 and i != c:
                        self._armsDelay[i] = self._armsDelay[i] + 1
                    # I cannot put it to zero or it seems like an unpulled arm
                    if d > self.maxDelay:
                        self._armsDelay[i] = self.maxDelay + 1
                    if i == c:
                        self._armsDelay[i] = 1

                # States update
                self.compute_states()

                # Additional termination condition due to finite horizon
                if t == horizon - 1:
                    return result
                t = t + 1
        return result

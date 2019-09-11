''' Environment for a multi-armed bandit problem'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.environment.Environment import Environment
from adaptiveRank.Results import *
from adaptiveRank.tools.io import c_print
from adaptiveRank.arm.Bernoulli import Bernoulli

from numpy import arange, around, array, linspace, unique, zeros
from random import seed, randint
import sortednp as snp

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, horizon, nbBuckets, gamma, fraTop, maxDelay, approximate, modality, policy_name):
        self.horizon = horizon
        self.nbBuckets = nbBuckets
        self.gamma = gamma
        self.fraTop = fraTop
        self.maxDelay = maxDelay
        self._approximate = approximate # Specifies whether to use binary rewards or not
        self._modality = modality # Specifies the learning problem: 0 full, 1 arm ordering, 2 rank estimation
        self.policy_name = policy_name
        self.arms = [] # List of Bernoulli objects
        self._meanArms = []
        self.nbArms = 0
        self.r_star = 0


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

        # Result data structure initialization
        result = Result(horizon)
        t = 0

        # Arm Creation
        self.nbArms = self._arm_creation(nbRepetition)
        self._armsDelay = zeros(self.nbArms)
        self._armsIndexes = arange(self.nbArms)
        assert self._armsIndexes[-1] == self.nbArms -1, "Wrong arm creation"
        c_print(4, "MAB.py, INIT: Arm Indexes {} Binary Rewards {}".format(self._armsIndexes, self._approximate))
        self._armsStates = zeros(self.nbArms) # Expected rewards of each arm according to suffered delays
        self.r_star = self._r_star_computation()

        result.setNbArms(self.nbArms)

        # Learning Modality Message Passing
        if self._modality == 2: # Rank Estimation 
            if self.policy_name in ['PI Low', 'PI ucb']:
                policy.overwriteArmMeans(self._meanArms)

        while t < horizon:
            c_print(1, "\n===\nMAB.py, play(): round {}\n===".format(t))

            if t == 0:
                self.compute_states()
                c_print(1, "MAB.py, play(): current delays: {}".format(self._armsDelay))
                c_print(1, "MAB.py, play(): arm states: {}".format(self._armsStates))
                # RStar policy update
                if self.policy_name == "Ghost":
                    policy.initialize(self.r_star)

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


    def _arm_creation(self, seed_init):
        seed(seed_init)
        starting_grid = linspace(0.0, 1.0, self.nbBuckets, endpoint = True)
        c_print(1, "Buckets: {}".format(starting_grid))
        delta = 1.0/(self.nbBuckets) # Previously adopted delta
        new_extreme = delta*self.fraTop*self.nbBuckets
        good_arms = linspace(0.0, new_extreme, self.nbBuckets, endpoint = False)
        c_print(1, "Good arms: {}".format(good_arms))
        means = around(array(snp.merge(starting_grid, good_arms)), 2) # evenly round to 2 decimals 
        means = 1 - unique(means)
        self._meanArms = means
        c_print(4, "\n=========MAB_INIT=========")
        c_print(4, "MAB.py, Arm means: {}".format(means))
        for mu in means:
            delay = randint(3, self.maxDelay)
            tmpArm = Bernoulli(mu, self.gamma, delay, self._approximate)
            self.arms.append(tmpArm)
            c_print(4, "MAB INIT: Created arm: {}".format(tmpArm))
        return len(means)


    def _r_star_computation(self):
        avgs = [self._avg(i) for i in  np.arange(1,len(self.arms),1)]
        r_star = array(avgs).argmax()
        c_print(4, "MAB.py, ARM CREATION Obtained avgs: {}, r_star: {}".format(avgs, r_star))
        return r_star


    def _avg(self, r):
        c_print(1, r)
        delayed_means = [arm.computeState(r) for arm in self.arms]
        c_print(1, "First {} arm means: {}".format(r, delayed_means[:r]))
        partial_sum = sum(delayed_means[:r])
        avg = partial_sum / (r)
        c_print(1, "Partial Sum {}, Average {}. over {} arms".format(partial_sum, avg, r))
        return avg


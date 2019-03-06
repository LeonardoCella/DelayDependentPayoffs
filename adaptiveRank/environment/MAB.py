''' Environment for a multi-armed bandit problem'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

from adaptiveRank.environment.Environment import Environment
from adaptiveRank.Results import *
from adaptiveRank.tools.io import c_print
from adaptiveRank.arm.Bernoulli import Bernoulli

from numpy import linspace, around, array

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, horizon, nbarms, nRepetitions, gamma, max_delay):
        self.horizon = horizon
        self.nbArms =  nbarms
        self.gamma = gamma
        self.max_delay = max_delay
        self._nbRepetitions = nRepetitions

        # Arm Creation
        means = linspace(0.0, 1.0, self.nbArms, endpoint = False) 
        c_print(2, "Arm default means: {}".format(means))
        arms = []
        means = around(array(means), 3)
        for mu in means:
            tmp_arm = Bernoulli(mu, self.gamma, self.max_delay)
            arms.append(tmp_arm)
            c_print(2, "MAB: Created arm: {}".format(tmp_arm))

    def play(self, policy, horizon, nbRepetition):
        ''' Called once per policy from __init__ of Evaluation. Is the one that manages the rounds.'''

        policy.startGame()
        result = Result(self.nbArms, horizon)

        c_print(1, "MAB PARAMETERS: nbArms: {}, horizon: {}, dimension: {}, spanned_d: {}".format(self.nbArms, self.horizon, self.d, self.sd))

        for t in range(horizon):
            c_print(1, "MAB play() simulation: before pol.play()")
            choice = policy.choice(context_vectors = self._contexts[t,:self.nbArms,...], t = t)
            c_print(1, "MAB play() simulation: theta star shape: {}".format(self._theta_star.shape))
            c_print(1, "MAB play() simulation: contexts shape: {}".format(self._contexts[t,:self.nbArms,...].shape))
            c_print(1, "MAB play() simulation: product shape: {}".format(self._contexts[t,:self.nbArms,...].dot(self._theta_star).shape))
            best_choice = np.argmax(self._rewards[nbRepetition, t,:])
            c_print(1, "MAB play() simulation: argmax {}".format(best_choice))
            mu = np.inner(self._theta_star , self._contexts[t,choice,:])
            best_mu = np.inner(self._theta_star, self._contexts[t,best_choice,:])
            c_print(1, "MAB play() simulation: best_mu {}, mu:{}".format(best_mu, mu))

            reward = self._rewards[nbRepetition, t,choice]
            max_reward = self._rewards[nbRepetition, t,best_choice]
            # Wrong Prediction
            if choice != best_choice:
                c_print(1, "MAB play() simulation: choice {} and best choice {}".format(choice, best_choice))
                result.storeWrongRound(t)
            # Policy update with the bandit feedback
            policy.getReward(self._contexts[t,choice,:], reward)
            result.store(t, choice, reward)
            best_expection += max_reward

        result.storeBestExpectation(best_expection)
        c_print(4, "End Iteration over rounds")

        return result

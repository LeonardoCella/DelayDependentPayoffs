''' Environment for a multi-armed bandit problem'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

from soful.environment.Environment import Environment
from soful.Results import *
from soful.tools.io import controlled_print

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, datasetID, dimension, reduction_ratio, alterate_reduction, max_horizon, nbarms, nRepetitions):
        self.horizon = max_horizon
        self.nbArms =  nbarms
        self.red = reduction_ratio
        self.red_alt = alterate_reduction
        # Dimensions adopted for the dataset generation
        self.d = dimension
        self.sd_policy = self.d - int(self.d * self.red)
        self.sd = self.d - int((self.red - self.red_alt) * self.d)
        self._datasetID = datasetID
        self._nbRepetitions = nRepetitions

        self._DATA = {0:'Synthetic', 1:'Bank', 2:'Spam', 3:'MFeat', 4:'Pendigits', 5:'CMC', 6:'SatImage', 7:'Covtype', 8:'Led', 9:'Vehicle'}
        self.dataset = self._DATA[self._datasetID]
        
        # Synthetic generated Dataset
        if self._datasetID == 0:
            self._contexts = self._preload()
            self._theta_star = self._parameter_loading()

            self._rewards = np.ndarray(shape=(nRepetitions, horizon+1, nbarms))
            for i in range(nRepetitions):
                self._rewards[i] = self._generate_rewards(self._contexts, self._theta_star)[:,:self.nbArms]
        else:# Real Datasets
            if self._datasetID == 1:
                from soful.arm.Bank import Bank
                self._dataset_handler = Bank(reduction_ratio)
            elif self._datasetID == 2:
                from soful.arm.Spam import Spam
                self._dataset_handler = Spam(reduction_ratio)
            elif self._datasetID == 3:
                from soful.arm.Mfeat import Mfeat 
                self._dataset_handler = Mfeat(reduction_ratio)
            elif self._datasetID == 4:
                from soful.arm.Pendigits import Pendigits
                self._dataset_handler = Pendigits(reduction_ratio)
            elif self._datasetID == 5:
                from soful.arm.CMC import CMC
                self._dataset_handler = CMC(reduction_ratio)
            elif self._datasetID == 6:
                from soful.arm.SatImage import SatImage
                self._dataset_handler = SatImage(reduction_ratio) 
            elif self._datasetID == 7:
                from soful.arm.Covtype import Covtype
                self._dataset_handler = Covtype(reduction_ratio)
            elif self._datasetID == 8:
                from soful.arm.Led import Led
                self._dataset_handler = Led(reduction_ratio)
            elif self._datasetID == 9:
                from soful.arm.Vehicle import Vehicle
                self._dataset_handler = Vehicle(reduction_ratio)
            self.horizon = self._dataset_handler.get_nB_rounds()
            self.nbArms = self._dataset_handler.get_nB_arms()
            self.d = self._dataset_handler.get_d()
            self.sd_policy = self.d - int(self.d * reduction_ratio)
            self.sd = self.d - int(self.d * self.red)

        if self.horizon > max_horizon:
            self.horizon = max_horizon
        controlled_print(1, "==={}===".format(self.dataset))
        controlled_print(4, "MAB INIT: K {}, T {}, d {}".format(self.nbArms, self.horizon, self.d))

    def play(self, policy, horizon, nbRepetition):
        ''' Called once per policy from __init__ of Evaluation. Is the one that manages the rounds.'''

        policy.startGame()
        result = Result(self.nbArms, horizon)

        controlled_print(1, "MAB PARAMETERS:nbArms: {}, horizon: {}, dimension: {}, spanned_d: {}".format(self.nbArms, self.horizon, self.d, self.sd))
        
        if self._datasetID != 0:
            self._dataset_handler.reset()

            # Reshuffle of the dataset_handler. I pass the nbRepetition since is necessary to define the seed.
            self._dataset_handler.shuffle(nbRepetition)
       
        best_expection = 0
        for t in range(horizon):

            if self._datasetID == 0:
                controlled_print(1, "MAB play() simulation: before pol.play()")
                choice = policy.choice(context_vectors = self._contexts[t,:self.nbArms,...], t = t)
                controlled_print(1, "MAB play() simulation: theta star shape: {}".format(self._theta_star.shape))
                controlled_print(1, "MAB play() simulation: contexts shape: {}".format(self._contexts[t,:self.nbArms,...].shape))
                controlled_print(1, "MAB play() simulation: product shape: {}".format(self._contexts[t,:self.nbArms,...].dot(self._theta_star).shape))
                best_choice = np.argmax(self._rewards[nbRepetition, t,:])
                controlled_print(1, "MAB play() simulation: argmax {}".format(best_choice))
                mu = np.inner(self._theta_star , self._contexts[t,choice,:])
                best_mu = np.inner(self._theta_star, self._contexts[t,best_choice,:])
                controlled_print(1, "MAB play() simulation: best_mu {}, mu:{}".format(best_mu, mu))

                reward = self._rewards[nbRepetition, t,choice]
                max_reward = self._rewards[nbRepetition, t,best_choice]
                # Wrong Prediction
                if choice != best_choice:
                    controlled_print(1, "MAB play() simulation: choice {} and best choice {}".format(choice, best_choice))
                    result.storeWrongRound(t)
                # Policy update with the bandit feedback
                policy.getReward(self._contexts[t,choice,:], reward)
            else: # Real dataset!
                contexts_by_round = self._dataset_handler.get_round_contexts()
                controlled_print(1, "MAB play() real_data: contexts type {} shape: {}".format(type(contexts_by_round), contexts_by_round.shape))
                controlled_print(1, "MAB play() real_data: contexts {}".format(contexts_by_round))
                choice = policy.choice(context_vectors = contexts_by_round, t = t)
                best_choice = self._dataset_handler.get_best_arm() 
                if choice == best_choice:
                    reward = 1
                    controlled_print(1, "MAB play() real_data: choice {} and best choice: {}".format(choice, best_choice))
                else:
                    controlled_print(1, "MAB play() real_data: choice {} and best choice: {}".format(choice, best_choice))
                    reward = 0
                    result.storeWrongRound(t)
                max_reward = 1
                policy.getReward(contexts_by_round[choice], reward)
                controlled_print(1, "MAB play() context_chosen shape: {}".format(contexts_by_round[choice].shape))
            controlled_print(1, "MAB: Rewards *{} #{}".format(max_reward, reward))
            result.store(t, choice, reward)
            best_expection += max_reward

        result.storeBestExpectation(best_expection)
        controlled_print(4, "End Iteration over rounds")

        return result

    def _preload(self):
        if self._datasetID == 0:
            from soful.arm.Context_Generator import Context_Generator
            context_generator = Context_Generator(self.d, self.sd, self.horizon)
            return context_generator._contexts

    def _parameter_loading(self):
        if self._datasetID == 0:
            from soful.arm.Context_Generator import Context_Generator
            context_generator = Context_Generator(self.d, self.sd, self.horizon)
            return context_generator._hidden_parameter

    def _generate_rewards(self, contexts, theta):
        controlled_print(1, "MAB: generate_rewards() contexts {}".format(contexts[1,2:4,...]))
        real_rewards = np.dot(contexts, theta)
        noise =  np.random.rand(real_rewards.shape[0], real_rewards.shape[1]) - 0.5
        noisy_rewards = real_rewards + noise
        controlled_print(1, "MAB: noisy_rewards :{} MAX: {} min {}".format(noisy_rewards[:3,:3], np.max(noisy_rewards), np.min(noisy_rewards)))
        return noisy_rewards

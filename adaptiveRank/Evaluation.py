'''Utility class for the performance evaluation'''

__author__ = "Leonardo Cella"
__version__ = "0.1"

import numpy as np
from adaptiveRank.tools.io import c_print
from joblib import Parallel, delayed

def parallel_repetitions(evaluation, policy, horizon, i):
    c_print(4, "\nEVALUATION: parallel_repetition index {}, T: {}".format(i+1, horizon))
    result = evaluation.environment.play(policy, horizon, i)
    return (i,result)

class Evaluation:
    def __init__(self, env, pol, nbRepetitions, index, n_job_ite):
        ''' Initialized in the run.py file.'''
        self.environment = env
        self.policy = pol

        self.horizon = env.horizon
        self.index = index #Policy Index
        self.nbArms = env.nbArms
        self.nbRepetitions = env._nbRepetitions
        self.nbPulls = np.zeros((self.nbRepetitions, self.nbArms))

        self.rewards = np.zeros(self.nbRepetitions)
        self.cumSumRwd = np.zeros((self.nbRepetitions, self.horizon))

        c_print(4,"===EVALUATION INIT: horizon {}, nbArms {}, nbRepetitions {}".format(self.horizon, self.nbArms, self.nbRepetitions))

        # Parallel call to the policy run over the number of repetitions. I got the final results. TO BE TESTED!!!
        with Parallel(n_jobs = n_job_ite) as parallel:
            repetitionIndex_results = parallel(delayed(parallel_repetitions)(self, pol, self.horizon, i) for i in range(nbRepetitions))
        #repetitionIndex_results = Parallel(n_jobs=n_job_ite)(delayed(parallel_repetitions) (self, pol, self.horizon, i) for i in range(nbRepetitions))
        for i, result in repetitionIndex_results:
            self.nbPulls[i,:] = result.getNbPulls()
            self.rewards[i] = result.getReward() # Over the flattened array
            self.cumSumRwd[i] = result.getCumSumRwd()
            c_print(1, "EVALUATION_INDEX {} ITERATION {}/{}  best_Expectation {} - rewards {}".format(self.index, i+1, nbRepetitions, result._bestExpectation, self.rewards[i]))
            c_print(1, "EVALUATION_INDEX {} ITERATION {}/{} regret: {}".format(self.index, i+1, nbRepetitions, self.regrets[i]))

        c_print(4, "End iteration over repetitions")

        # Averaged best Expectation.
        self.meanBestExpectation = np.mean(self.arr_bestExpectation)
        self.meanRegret = np.mean(self.regrets)
        self.meanReward = np.mean(self.rewards)

    def cumSumRwds(self):
        return self.cumSumRwd

    def meanNbDraws(self):
        return np.mean(self.nbPulls, 0)

    def getRegrets(self):
        return self.regrets

    def getMeanRegret(self):
        return self.meanRegret

    def getRewards(self):
        return self.rewards

    def getMeanReward(self):
        return self.meanReward

    def getWrongRounds(self):
        return self.wrong_rounds

    def getCompressedRounds(self):
        return self.compressed_rounds

    def getBestExpectations(self):
        return self.arr_bestExpectation

    def getMeanBestExpectation(self):
        return self.meanBestExpectation

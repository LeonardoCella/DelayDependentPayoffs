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
    def __init__(self, env, pol, horizon, policyName, nbRepetitions ):
        ''' Initialized in the run.py file.'''
        # Associated learning problem: policy, environment
        self.environment = env
        self.policy = pol

        # Learning problem parameters: horizon, policy name, nbRepetitions 
        self.horizon = horizon
        self.polName = policyName
        self.nbRepetitions = nbRepetitions

        # Data Structurs to store the results of different reward samples
        self.rewards = np.zeros(self.nbRepetitions)
        self.cumSumRwd = np.zeros((self.nbRepetitions, self.horizon))

        c_print(4,"===Evaluation.py, INIT: {} over {} rounds for {} nbRepetitions".format(self.polName, self.horizon, self.nbRepetitions))

        # Parallel call to the policy run over the number of repetitions
        with Parallel(n_jobs = self.nbRepetitions) as parallel:
            repetitionIndex_results = parallel(delayed(parallel_repetitions)(self, self.policy, self.horizon, i) for i in range(nbRepetitions))

        # Results extrapolation
        for i, result in repetitionIndex_results:
            self.rewards[i] = result.getReward() # Over the flattened array
            self.cumSumRwd[i] = result.getCumSumRwd()

        c_print(2, "End iteration over repetitions")

        # Averaged best Expectation.
        self.meanReward = np.mean(self.rewards)
        self.meanCumSumRwd = np.mean(self.cumSumRwd)

    def cumSumRwds(self):
        return self.cumSumRwd

    def getRewards(self):
        return self.rewards

    def getMeanReward(self):
        return self.meanReward

    def getMeanCumSumRwd(self):
        return self.meanCumSumRwd

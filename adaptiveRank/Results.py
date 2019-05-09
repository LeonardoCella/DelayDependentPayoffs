'Utility class that manages the results of MAB experiments'

import numpy as np
from adaptiveRank.tools.io import c_print


class Result:
    """Class that analyzes the outcome of a bandit experiment"""

    def __init__(self, horizon):
        # Initially all the rounds have no choices or rewards.
        self.choices = np.zeros(horizon, dtype=np.int)
        self.rewards = np.zeros(horizon)

    # Store the info for round t
    def store(self, t, choice, reward):
        self.choices[t] = choice
        self.rewards[t] = reward

    def getCumSumRwd(self):
        return np.cumsum(self.rewards)

    def getReward(self):
        return np.sum(self.rewards)

    def __repr__(self):
        return "<Result choices:%s \n reward %s>" % (self.choices, self.rewards)

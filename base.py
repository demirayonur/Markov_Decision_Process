# Includes the base MDP class


import numpy as np


class MDP:

    def __init__(self, n_state, n_action, sensitivity=1e-7):

        """
        Construction of the base class

        :param n_state: Integer, Number of states
        :param n_action: Integer, Number of actions
        :param sensitivity: Float, Optional --> Sensitivity in the computational issues.

        @CAUTION 2: For now, we don't have any solve function since it is not clear
        whether we have finite or infinite horizons.

        """

        self.n_state = n_state
        self.n_action = n_action
        self.states = list(range(n_state))
        self.actions = list(range(n_action))
        self.sensitivity = sensitivity

        # Let's define empty value functions and policy

        self.values = np.empty(shape=self.n_state)  # 1D Numpy Array
        self.policy = {}  # dictionary, (state, action)

    def transition_prob_check(self):

        pass

    def reward_function_check(self):

        pass

    def check(self):

        self.transition_prob_check()
        self.reward_function_check()

    def solve(self):

        pass

    def plot_policy(self):

        pass
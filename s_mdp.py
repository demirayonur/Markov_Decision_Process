# This module includes SMDP class
# which responsible for stationary
# MDP models. It is inherited
# from base MDP class


from base import MDP
import numpy as np


class SMDP(MDP):

    def __init__(self, transition_prob, rewards, n_state, n_action):

        """
        Construction of the Stationary MDPs

        :param transition_prob: 3D Array
        :param rewards: 2D Array
        :param n_state: Integer
        :param n_action: Integer
        """

        MDP.__init__(self, n_state=n_state, n_action=n_action)
        self.transition_probs = transition_prob
        self.rewards = rewards

    def transition_probability_check(self):

        """
        Checks whether given transition probability meets the
        dimensionality requirements of given state and action
        numbers. It also checks whether for any given SxA, sum
        of probabilities of states s' equal 1 or not.

        :return: % RAISE %
        """

        s_from, action, s_to = self.transition_probs.shape

        if s_from != self.n_state or s_to != self.n_state:
            raise ValueError('Invalid state number')

        if action != self.n_action:
            raise ValueError('Invalid action number')

        sum_ = [np.sum(self.transition_probs[s, a, :]) for s in self.states for a in self.actions]
        cond = [i for i in sum_ if i < 1 - self.sensitivity or i > 1 + self.sensitivity]
        if len(cond) > 0:
            raise ValueError('Not a stochastic tensor')

    def reward_function_check(self):

        """
        Checks whether given reward function has appropriate
        dimensionality or not based on the given state
        and action numbers

        :return: % RAISE %
        """

        s, a = self.rewards.shape

        if s != self.n_state:
            raise ValueError('Invalid state number')

        if a != self.n_action:
            raise ValueError('Invalid action number')

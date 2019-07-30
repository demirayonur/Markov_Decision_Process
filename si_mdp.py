# This module includes SIMDP class
# which responsible for stationary
# Infinite-Horizon MDP models. It is
# inherited from SMDP class

import utils_ as utils
from s_mdp import SMDP
import numpy as np


class SIMDP(SMDP):

     def __init__(self, transition_prob, rewards, n_state, n_action, discount_factor):

         SMDP.__init__(self, transition_prob, rewards, n_state, n_action)
         self.discount_factor = discount_factor

     def value_iteration(self):

         """
         Performs value iteration and finds the optimal policy.

         :return: % INPLACE % + Returns the history

         Caution: History means that evaluation of value function over time
         """

         history = [[] for s in range(self.n_state)]  # For plotting

         self.values = np.zeros(shape=self.n_state)

         # Compute optimal value function

         while True:

             delta = 0

             # For plotting
             for s in range(self.n_state):
                 history[s].append(self.values[s])

             for s in range(self.n_state):
                 s_prev = self.values[s]
                 self.values[s] = self.discount_factor * max([self.rewards[s, a] +
                                                         np.sum(self.transition_probs[s, a, :] * self.values[:])
                                                         for a in range(self.n_action)])
                 delta = max(delta, abs(self.values[s] - s_prev))

             if delta < 1e-6:
                 break

         # Extract the optimal policy from the optimal values.

         for s in range(self.n_state):
             self.policy[s] = np.argmax(np.array([np.sum(self.transition_probs[s, a] * self.values[:])
                                                  for a in range(self.n_action)]))

         return history  # return the history for further plotting options

     def solve(self):

         history = self.value_iteration()
         return history

     def plot_policy(self):

         utils.plot_si_policy(self.policy, self.n_state, self.n_action)

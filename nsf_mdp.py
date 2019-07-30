# This module includes NSFMDP class
# which responsible for non-stationary
# Finite-Horizon MDP models. It is
# inherited from SMDP class


from ns_mdp import NSMDP
import numpy as np


class NSFMDP(NSMDP):

    def __init__(self, transition_prob, rewards, n_state, n_action, decision_epochs, last_vals):

        """

        :param decision_epochs: Integer
        :param last_vals: 1D Array
        """
        NSMDP.__init__(self, transition_prob, rewards, n_state, n_action)
        self.last_vals = last_vals
        self.num_dec = decision_epochs

        # Let's define empty value functions and policy

        self.values = np.empty(shape=(self.num_dec, self.n_state))  # 2D Numpy Array
        self.policy = {}  # dictionary, ((time, state), action)

        self.check()

    def check(self):

        self.transition_prob_check()
        self.reward_function_check()

        if len(self.last_vals) != self.n_state:
            raise ValueError('There is a problem with the number of states in the last decision stage')

        if self.transition_probs.shape[0] != self.num_dec or self.rewards.shape[0] != self.num_dec:
            raise  ValueError('There is a problem with the number of decision epochs')

    def backward_induction(self):

        for s in self.states:
            self.values[self.num_dec - 1, s] = self.last_vals[s]

        for t in range(self.num_dec - 2, -1, -1):

            # Compute value functions

            for s in self.states:

                self.values[t, s] = max([
                    self.rewards[t, s, a] + np.sum(self.transition_probs[t, s, a, :] * self.values[t + 1, :])
                    for a in self.actions
                ])

            # Compute Policies

            for s in self.states:

                self.policy[t, s] = np.argmax(np.array([
                    self.rewards[t, s, a] + np.sum(self.transition_probs[t, s, a, :] * self.values[t + 1, :])
                    for a in self.actions
                ]))

    def solve(self):

        self.backward_induction()

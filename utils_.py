import matplotlib.pyplot as plt
import numpy as np


def plot_si_policy(policy, n_state, n_action):

    """
    Plots the given policy

    :param policy: Dictionary, (state, action)
    :param n_state: Integer
    :param n_action: Integer

    :return: % PLOT %
    """
    states = list(range(n_state))
    actions = list(range(n_action))

    plt.figure(figsize=(18, 5))
    plt.plot(states, [policy[state] for state in states], color='purple')
    plt.xlabel('states')
    plt.ylabel('actions')
    plt.xticks(np.arange(min(states), max(states) + 1, 1.0))
    plt.yticks(np.arange(min(actions), max(actions) + 1, 1.0))
    plt.title('Optimal Policy')
    plt.grid()
    plt.show()



def plot_history(history):

    """
    Plots the given history of value functions

    :param history: List of lists,

    :return: % PLOT %
    """

    plt.figure(figsize=(18, 5))

    for s in range(len(history)):
        name = 'state ' + str(s)
        plt.plot(list(range(len(history[s]))), history[s])
        plt.scatter(list(range(len(history[s]))), history[s], label=name, s=40)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        #plt.xticks(0, max(states) + 1, 1.0))
        plt.title('Evaluation of Value Functions for Each State')
        plt.legend()
    plt.show()


def create_random_simdp(n_state, n_action, gamma=0.1):

    """
    Creates random MDP environment based on the given
    number of states, actions and discount value

    :param n_state: Integer
    :param n_action: Integer
    :param gamma: Float, optional
    :return: Tuple <S, A, T, R, G>
    """

    def sum_1_array(n_size):

        array = np.random.randint(1, 10, n_size)
        total = np.sum(array)
        return array / total

    states = list(range(n_state))
    actions = list(range(n_action))
    discount_factor = gamma
    rewards = np.random.randint(1, 10, size=(n_state, n_action))
    transitions = np.empty(shape=(n_state, n_action, n_state))
    for i in states:
        for j in actions:
            transitions[i, j,] = sum_1_array(n_state)

    from si_mdp import SIMDP

    mdp = SIMDP(transitions, rewards, n_state, n_action, discount_factor)

    return mdp


def create_random_sfmdp(n_state, n_action, dec_ep):

    """
    Creates random MDP environment based on the given
    number of states, actions and discount value

    :param n_state: Integer
    :param n_action: Integer
    :param dec_ep: Integer
    :return: Tuple <S, A, T, R, N>
    """

    def sum_1_array(n_size):

        array = np.random.randint(1, 10, n_size)
        total = np.sum(array)
        return array / total

    states = list(range(n_state))
    actions = list(range(n_action))

    last_vals = np.zeros(shape=n_state)
    rewards = np.random.randint(1, 10, size=(n_state, n_action))
    transitions = np.empty(shape=(n_state, n_action, n_state))
    for i in states:
        for j in actions:
            transitions[i, j,] = sum_1_array(n_state)

    from sf_mdp import SFMDP

    mdp = SFMDP( transitions, rewards, n_state, n_action, dec_ep, last_vals)

    return mdp


def create_random_nsfmdp(n_state, n_action, dec_ep):

    """
    Creates random MDP environment based on the given
    number of states, actions and discount value

    :param n_state: Integer
    :param n_action: Integer
    :param dec_ep: Integer
    :return: Tuple <S, A, T, R, N>
    """

    def sum_1_array(n_size):

        array = np.random.randint(1, 10, n_size)
        total = np.sum(array)
        return array / total

    states = list(range(n_state))
    actions = list(range(n_action))

    last_vals = np.zeros(shape=n_state)
    rewards = np.random.randint(1, 10, size=(dec_ep, n_state, n_action))
    transitions = np.empty(shape=(dec_ep, n_state, n_action, n_state))

    for t in range(dec_ep):
        for i in states:
            for j in actions:
                transitions[t, i, j,] = sum_1_array(n_state)

    from nsf_mdp import NSFMDP

    mdp = NSFMDP( transitions, rewards, n_state, n_action, dec_ep, last_vals)

    return mdp
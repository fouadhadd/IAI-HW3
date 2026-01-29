from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np
import copy


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #
    U_final = None
    # TODO:
    # ====== YOUR CODE: ======
    U_temp = copy.deepcopy(U_init)
    delta = 0
    while True:
        U_final = copy.deepcopy(U_temp)
        delta = 0
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                state = (i, j)

                if mdp.board[i][j] == "WALL":
                    U_temp[i][j] = None
                    continue

                if state in mdp.terminal_states:
                    U_temp[i][j] = float(mdp.get_reward(state))

                else:
                    max_utility = float('-inf')
                    for action in mdp.actions:
                        utility = 0
                        probs = mdp.transition_function[action]
                        for prob, move in zip(probs, mdp.actions.keys()):
                            next_state = mdp.step(state, move)
                            utility += prob * U_final[next_state[0]][next_state[1]]

                        if utility > max_utility:
                            max_utility = utility

                    U_temp[i][j] = float(mdp.get_reward(state)) + mdp.gamma * max_utility

                delta = max(delta, abs(U_final[i][j] - U_temp[i][j]))

        if delta < (epsilon * ((1 - mdp.gamma) / mdp.gamma)):
            break
    # ========================
    return U_final


def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    policy = None
    # TODO:
    # ====== YOUR CODE: ====== 
    policy = [[None for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            state = (i, j)

            if mdp.board[i][j] == "WALL" or state in mdp.terminal_states:
                policy[i][j] = None

            else:
                max_arg = -1
                max_val = float('-inf')

                for action in mdp.actions:
                    val = 0
                    probs = mdp.transition_function[action]
                    for prob, move in zip(probs, mdp.actions.keys()):
                        next_state = mdp.step(state, move)
                        val += prob * (U[next_state[0]][next_state[1]])

                    if val > max_val:
                        max_arg = action
                        max_val = val

                policy[i][j] = max_arg
    # ========================
    return policy


def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    U = None
    # TODO:
    # ====== YOUR CODE: ======
    U_temp = [[0 for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    delta = 0
    while True:
        U = copy.deepcopy(U_temp)
        delta = 0
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                state = (i, j)

                if mdp.board[i][j] == "WALL":
                    U_temp[i][j] = None
                    continue

                if state in mdp.terminal_states:
                    U_temp[i][j] = float(mdp.get_reward(state))

                else:
                    utility = 0
                    probs = mdp.transition_function[Action(policy[i][j])]
                    for prob, move in zip(probs, mdp.actions.keys()):
                        next_state = mdp.step(state, move)
                        utility += prob * U[next_state[0]][next_state[1]]

                    U_temp[i][j] = float(mdp.get_reward(state)) + mdp.gamma * utility

                delta = max(delta, abs(U[i][j] - U_temp[i][j]))

        if delta == 0:
            break
    # ========================
    return U


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======
    optimal_policy = copy.deepcopy(policy_init)
    while True:
        U = policy_evaluation(mdp, optimal_policy)
        unchanged = True
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                state = (i, j)

                if mdp.board[i][j] == "WALL" or state in mdp.terminal_states:
                    optimal_policy[i][j] = None

                else:
                    max_arg = -1
                    max_val = float('-inf')

                    for action in mdp.actions:
                        val = 0
                        probs = mdp.transition_function[action]
                        for prob, move in zip(probs, mdp.actions.keys()):
                            next_state = mdp.step(state, move)
                            val += prob * (U[next_state[0]][next_state[1]])

                        if val > max_val:
                            max_arg = action
                            max_val = val

                    if max_arg != optimal_policy[i][j]:
                        optimal_policy[i][j] = max_arg
                        unchanged = False

        if unchanged:
            break
    # ========================
    return optimal_policy


def mc_algorithm(
        sim,
        num_episodes,
        gamma,
        num_rows=3,
        num_cols=4,
        actions=[Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT],
        policy=None,
):
    # Given a simulator, the number of episodes to run, the number of rows and columns in the MDP, the possible actions,
    # and an optional policy, run the Monte Carlo algorithm to estimate the utility of each state.
    # Return tthe utility of each sate.

    V = None

    # ====== YOUR CODE: ======
    V = np.zeros((num_rows, num_cols))
    returns_sum = np.zeros((num_rows, num_cols))
    returns_count = np.zeros((num_rows, num_cols))

    for episode in sim.replay(num_episodes=num_episodes):
        states = []
        rewards = []

        for state, reward, action, actual_action in episode:
            states.append(state)
            rewards.append(reward)

        G = 0

        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            reward = rewards[t]
            G = gamma * G + reward

            if state not in states[:t]:
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]

    V[1,1] = None

    # =========================

    return V

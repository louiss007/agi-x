"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2023/12/2 13:53
# @FileName : q_learning.py
# @Email    : quant_master2000@163.com
==========================
"""
import pandas as pd
import numpy as np
import time


class QLearning:
    """ Q-learning Algorithm """
    fresh_time = 0.3

    def __init__(self, n_states, epsilon, alpha, gamma, actions):
        self.n_states = n_states        # number of states
        self.actions = actions
        self.n_action = len(actions)    # number of actions
        self.alpha = alpha              # learning rate
        self.gamma = gamma              # discount factor
        self.epsilon = epsilon          # hyper para of ε-greedy
        self.q_table = self.init_q_table()

    def init_q_table(self):
        """initiate q table"""
        table = pd.DataFrame(np.zeros([self.n_states, self.n_action]), columns=self.actions)
        return table

    def show_env(self, s, episode, step):
        """show current environment"""
        env_list = ['-'] * (self.n_states - 1) + ['T']
        if s == 'terminal':  # whether is getting terminal
            interaction = 'Episode %s: total_steps = %s' % (episode + 1, step)
            print(f'\r{interaction}')
            time.sleep(0.5)
        else:
            env_list[s] = 'o'
            interaction = ''.join(env_list)
            print(f'\r{interaction}', end='')
            time.sleep(self.fresh_time)  # set moving speed interval 0.3s

    def take_action(self, state):
        state_actions = self.q_table.iloc[state, :]
        if (np.random.uniform() > self.epsilon) or (state_actions == 0).all():
            # explore
            action = np.random.choice(self.actions)  # random to choose action
        else:
            # exploit
            action = state_actions.idxmax()  # choose the action with max_q value
        return action

    def reward_fn(self, state, action):
        """Given current state and action, compute reward and next state"""
        if action == 'right':
            if state == self.n_states - 2:  # whether is terminal or not
                state_n = 'terminal'
                r = 1  # yes reward
            else:
                state_n = state + 1
                r = 0  # no reward
        else:
            r = 0  # no reward
            state_n = max(0, state-1)
        return state_n, r

    def update(self, s, a, r, s_n):
        """update the values of q table"""
        if s_n != 'terminal':
            td_error = r + self.gamma * self.q_table.iloc[s_n, :].max() - self.q_table.loc[s, a]
        else:
            td_error = r
        self.q_table.loc[s, a] += self.alpha * td_error
        return s_n

    def run(self, max_episodes):
        for episode in range(max_episodes):
            step = 0
            s = 0  # initial state
            is_done = False     # a sign whether current exploration is over
            self.show_env(s, episode, step)
            while not is_done:
                a = self.take_action(s)         # choose action
                s_n, r = self.reward_fn(s, a)   # compute reward
                s = self.update(s, a, r, s_n)   # update q table
                if s == 'terminal':
                    is_done = True
                step += 1
                self.show_env(s_n, episode, step)

        return self.q_table


def main():
    max_episodes = 10
    n_states = 6
    epsilon = 0.9
    alpha = 0.1
    gamma = 0.9
    actions = ['left', 'right']
    np.random.seed(7)
    q_agent = QLearning(n_states, epsilon, alpha, gamma, actions)
    q_table = q_agent.run(max_episodes)
    print('\r\nQ-table:\n')
    print(q_table)
    return q_table


if __name__ == '__main__':
    main()

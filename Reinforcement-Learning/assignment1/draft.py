'''

This includes my test runs to understand the data structure when attempting the problem.
Feel free to go through and try it yourself.

'''
import gym
import numpy as np
from lake_envs import *
# uncomment to check out stochastic/deterministic environments 
env = gym.make("Deterministic-4x4-FrozenLake-v0")
#env = gym.make("Stochastic-4x4-FrozenLake-v0")

'''
	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
'''


P = env.P
nA = 4
nS = 16
gamma = 0.9
policy = 2 * np.ones(nS, dtype='int')


for state in P:
    A = P[state]       
    for action in A:
        for prob, next_state, reward, terminal in A[action]:
            print('p(s_{}|s_{},a_{})={}, with reward {}'.format(next_state,state,action,prob,reward))

env.close()



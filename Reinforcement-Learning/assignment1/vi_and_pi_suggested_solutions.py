### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

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
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def custimise_reward(state, next_state, reward):
	if state == next_state:
		reward = 0
	else:
		if next_state in [5,7,11,12]: reward = reward - 10	
		elif next_state == 15: reward = 10
		else: reward = 1
	return reward

def policy_evaluation(P, nS, nA, policy, gamma, tol):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

	value_function = np.zeros(nS)	
	k = 1
	while k:
		k = k + 1
		prev_value_function = value_function
		for state in P:
			A = P[state]
			policy_action = policy[state] # extract action from policy 
			for prob, next_state, reward, terminal in A[policy_action]:
				reward = custimise_reward(state, next_state, reward)
				value_function[state] = reward + gamma * prob * prev_value_function[next_state]
		if np.max(np.abs(value_function - prev_value_function)) < tol: break

	return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
	new_policy = np.empty(nS,dtype=int)	
	Q_function = dict()
	for state in P:
		A = P[state]
		for action in A:
			for prob, next_state, reward, terminal in A[action]:
				reward = custimise_reward(state, next_state, reward)
				Q_function[state] = np.empty(nA)
				Q_function[state][action] = reward + gamma * prob * value_from_policy[next_state] 
		# extract optimal policy from state-action value Q
		new_policy[state] = np.argmax(Q_function[state]) #if Q value doesn't cahnge much, 
														 #its ranking doesn't change as well
														 #so policy doesn't change as well

	return new_policy


def policy_iteration(P, nS, nA, gamma, tol):
	"""Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value_from_policy: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""
	policy = np.zeros(nS,dtype=int)	
	k = 1
	waiting = 0
	while k: 
		k = k + 1
		prev_policy = policy
		value_from_policy = policy_evaluation(P, nS, nA, prev_policy, gamma, tol)
		policy = policy_improvement(P, nS, nA, value_from_policy, gamma)
		if k > 1000 or np.all(prev_policy == policy): # force quit after 1000 iterations
			waiting = waiting + 1
			if waiting > 5: break # wait for 5 more iterations 
		

	print('Optiaml policy is found after {} policy iteration'.format(k-1))
	return value_from_policy, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		Terminate value iteration when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS,dtype=int)
	Q_function = dict()
	k = 1
	while k: 
		prev_value_function = value_function
		k = k + 1
		for state in P:
			A = P[state]
			for action in A:
				for prob, next_state, reward, terminal in A[action]:
					reward = custimise_reward(state, next_state, reward)
					Q_function[state] = np.empty(nA)
					Q_function[state][action] = reward + gamma * prob * prev_value_function[next_state]
			# extract state value from state-action value Q
			value_function[state] = np.max(Q_function[state])
		if np.max(np.abs(value_function - prev_value_function)) < tol: 
			# extract policy from state-action value Q
			for state in Q_function:
				policy[state] = np.argmax(Q_function[state])
			break

	print('Optiaml policy is found after {} value iteration'.format(k-1))
	return value_function, policy

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
  	print("Episode reward: %f" % episode_reward)




# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

	# comment/uncomment these lines to switch between deterministic/stochastic environments
	#env = gym.make("Deterministic-4x4-FrozenLake-v0")
	env = gym.make("Stochastic-4x4-FrozenLake-v0")
	#env = gym.make("Deterministic-8x8-FrozenLake-v0")

	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=1, tol=1e-5)
	print(p_pi);print(V_pi)
	#render_single(env, p_pi, 10)

	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=1, tol=1e-5)
	print(p_vi)
	V_vi = [round(V_vi[i],2) for i in range(env.nS)]
	print(V_vi)
	render_single(env, p_vi, 10)



import gym
from lake_envs import *
#env = gym.make('FrozenLake-v0') 
#env = gym.make("Deterministic-8x8-FrozenLake-v0")
# uncomment to check out stochastic/deterministic environments 
env = gym.make("Deterministic-4x4-FrozenLake-v0")
#env = gym.make("Stochastic-4x4-FrozenLake-v0")


for i_episode in range(20):
    observation = env.reset() # reset env when generating new episodes
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample() # take a random action
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

print(env.action_space) 
print(env.observation_space) 


env.close()

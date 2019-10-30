import gym
import argparse
from config import get_config
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole', 'pendulum', 'cheetah'])
parser.add_argument('--baseline', dest='use_baseline', action='store_true')
parser.add_argument('--no-baseline', dest='use_baseline', action='store_false')
parser.set_defaults(use_baseline=True)

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.env_name, args.use_baseline)
    env = gym.make(config.env_name)

    actionDiscrete = isinstance(env.action_space, gym.spaces.Discrete)
    print(actionDiscrete) # whether action is discrete
    print(env.observation_space.shape[0])
    print(env.action_space.n) if actionDiscrete else print(env.action_space.shape[0])
     

    env.close()
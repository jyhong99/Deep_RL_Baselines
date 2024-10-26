import gym
from baselines.agent.policy_based.on_policy.ppo import PPO
from baselines.agent.policy_based.off_policy.ddpg import DDPG

env_name = 'HalfCheetah-v4'
env = gym.make(env_name)
agent = PPO(env, reward_norm=True, adv_norm=True)
agent.train(env_name, normalized_env=True, max_iters=1000000, n_runners=8, runner_iters=10, eval_intervals=10000)
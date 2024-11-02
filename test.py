import gym
from baselines.agent.policy_based.on_policy.ppo import PPO
from baselines.agent.policy_based.off_policy.sac import SAC

env_name = 'HalfCheetah-v4'
env = gym.make(env_name)
agent = SAC(env, reward_norm=True, adaptive_alpha_mode=True, prioritized_mode=True)
agent.train(env_name,  max_iters=1000000, n_runners=8, runner_iters=100, eval_intervals=10000)
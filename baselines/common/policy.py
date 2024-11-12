import gym
import torch
import numpy as np
from copy import deepcopy
from baselines.common.buffer import RolloutBuffer, ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.train import Trainer, DistributedTrainer


class OnPolicyAlgorithm:
    def __init__(self,
        env, actor_size, critic_size, actor_activation, critic_activation, buffer_size, update_after, 
        actor_lr, critic_lr, gamma, lmda, vf_coef, ent_coef, reward_norm, adv_norm
    ):
        
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_dim = self.env.action_space.shape[0]
            self.action_type = 'continuous'
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.action_type = 'discrete'
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.action_type = 'multidiscrete'
            self.action_dim = self.env.action_space.nvec 


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timesteps = 0
        self.epsilon = 1e-8
        
        self.actor_size = actor_size
        self.critic_size = critic_size
        self.actor_activation = actor_activation
        self.critic_activation = critic_activation
        self.buffer_size = buffer_size
        self.update_after = update_after

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lmda = lmda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.reward_norm = reward_norm
        self.adv_norm = adv_norm
        
        self.buffer = RolloutBuffer(
            state_dim=self.state_dim, 
            action_dim=self.action_dim, 
            buffer_size=self.buffer_size, 
            device=self.device, 
            reward_norm=self.reward_norm, 
            epsilon=self.epsilon
            )

    @torch.no_grad()
    def act(self, *args, **kwargs):
        raise NotImplementedError()
    
    def GAE(self, values, next_values, rewards, dones):
        delta = rewards + (1. - dones) * self.gamma * next_values - values
        rets, advs = torch.clone(rewards), torch.clone(delta)
        for i in reversed(range(len(rets) - 1)):
            rets[i] += (1. - dones[i]) * self.gamma * rets[i + 1]
            advs[i] += (1. - dones[i]) * self.gamma * self.lmda * advs[i + 1]
        
        if self.adv_norm:
            advs = (advs - advs.mean()) / (advs.std() + self.epsilon)

        return rets, advs
   
    def learn(self, *args, **kwargs):
        raise NotImplementedError()

    def train(self, project_name, **config):
        self.eval_env = deepcopy(self.env)       
        self.project_name = config.get('project_name', project_name)
        self.load_path = config.get('load_path', None)
        self.seed = config.get('seed', 0)
        self.normalized_env = config.get('normalized_env', False)
        self.max_iters = config.get('max_iters', int(1e+6))
        self.n_runners = config.get('n_runners', 1)
        self.runner_iters = config.get('runner_iters', 10)
        self.eval_mode = config.get('eval_mode', True)
        self.eval_intervals = config.get('eval_intervals', 1000)
        self.eval_iters = config.get('eval_iters', 10)
        self.policy_type = 'on_policy'
        self.show_stats = config.get('show_stats', True)
        self.show_graphs = config.get('show_graphs', True)
    
        if self.n_runners > 1:
            self.trainer = DistributedTrainer(
                env=self.env, 
                eval_env=self.eval_env, 
                agent=self, 
                seed=self.seed, 
                )
            
            self.trainer.train(
                project_name=self.project_name, 
                load_path=self.load_path, 
                normalized_env=self.normalized_env,
                max_iters=self.max_iters, 
                n_runners=self.n_runners, 
                runner_iters=self.runner_iters, 
                eval_mode=self.eval_mode, 
                eval_intervals=self.eval_intervals, 
                eval_iters=self.eval_iters, 
                policy_type=self.policy_type,
                action_type=self.action_type,
                show_stats=self.show_stats, 
                show_graphs=self.show_graphs,
                )
        else:
            self.trainer = Trainer(
                env=self.env, 
                eval_env=self.eval_env, 
                agent=self, 
                seed=self.seed, 
                )
            
            self.trainer.train(
                project_name=self.project_name, 
                load_path=self.load_path, 
                normalized_env=self.normalized_env,
                max_iters=self.max_iters, 
                eval_mode=self.eval_mode, 
                eval_intervals=self.eval_intervals, 
                eval_iters=self.eval_iters, 
                show_stats=self.show_stats, 
                show_graphs=self.show_graphs,
                )

    def step(self, state, action, reward, next_state, done):
        result = None
        self.buffer.store(state, action, reward, next_state, done)
        if self.buffer.size >= self.update_after:
            states, actions, rewards, next_states, dones = self.buffer.sample() 
            result = self.learn(states, actions, rewards, next_states, dones)

        return result
                    
    def save(self, *args, **kwargs):
        raise NotImplementedError()

    def load(self, *args, **kwargs):
        raise NotImplementedError()








class OffPolicyAlgorithm:
    def __init__(self,
        env, actor_size, critic_size, actor_activation, critic_activation,
        buffer_size, batch_size, update_after, actor_lr, critic_lr, gamma, tau, reward_norm, 
        prioritized_mode, prio_alpha, prio_beta, prio_eps,
    ):

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_dim = self.env.action_space.shape[0]
            self.action_type = 'continuous'
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.action_type = 'discrete'
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.action_type = 'multidiscrete'
            self.action_dim = self.env.action_space.nvec 

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timesteps = 0
        self.epsilon = 1e-8

        self.actor_size = actor_size
        self.critic_size = critic_size
        self.actor_activation = actor_activation
        self.critic_activation = critic_activation
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_after = update_after

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma 
        self.tau = tau
        self.reward_norm = reward_norm

        self.prioritized_mode = prioritized_mode
        self.prio_alpha = prio_alpha
        self.prio_beta = prio_beta
        self.prio_eps = prio_eps

        if prioritized_mode: 
            self.buffer = PrioritizedReplayBuffer(
                state_dim=self.state_dim, 
                action_dim=self.action_dim, 
                buffer_size=self.buffer_size, 
                batch_size=self.batch_size,
                device=self.device, 
                alpha=self.prio_alpha,
                reward_norm=self.reward_norm, 
                epsilon=self.epsilon
                )
        else: 
            self.buffer = ReplayBuffer(
                state_dim=self.state_dim, 
                action_dim=self.action_dim, 
                buffer_size=self.buffer_size, 
                batch_size=self.batch_size,
                device=self.device, 
                reward_norm=self.reward_norm, 
                epsilon=self.epsilon
                )
        
    @torch.no_grad()
    def act(self, *args, **kwargs):
        raise NotImplementedError()
    
    def random_action(self):
        if self.action_type == 'continuous':
            return np.random.uniform(-1., 1., self.action_dim)
        else:
            return self.env.action_space.sample()

    def learn(self, *args, **kwargs):
        raise NotImplementedError()

    def train(self, project_name, **config):
        self.eval_env = deepcopy(self.env)       
        self.project_name = config.get('project_name', project_name)
        self.load_path = config.get('load_path', None)
        self.seed = config.get('seed', 0)
        self.normalized_env = config.get('normalized_env', False)
        self.max_iters = config.get('max_iters', int(1e+6))
        self.n_runners = config.get('n_runners', 1)
        self.runner_iters = config.get('runner_iters', 10)
        self.eval_mode = config.get('eval_mode', True)
        self.eval_intervals = config.get('eval_intervals', 1000)
        self.eval_iters = config.get('eval_iters', 10)
        self.policy_type = 'off_policy'
        self.show_stats = config.get('show_stats', True)
        self.show_graphs = config.get('show_graphs', True)
    
        if self.n_runners > 1:
            self.trainer = DistributedTrainer(
                env=self.env, 
                eval_env=self.eval_env, 
                agent=self, 
                seed=self.seed, 
                )
            
            self.trainer.train(
                project_name=self.project_name, 
                load_path=self.load_path, 
                normalized_env=self.normalized_env,
                max_iters=self.max_iters, 
                n_runners=self.n_runners, 
                runner_iters=self.runner_iters, 
                eval_mode=self.eval_mode, 
                eval_intervals=self.eval_intervals, 
                eval_iters=self.eval_iters, 
                policy_type=self.policy_type,
                action_type=self.action_type,
                show_stats=self.show_stats, 
                show_graphs=self.show_graphs,
                )
        else:
            self.trainer = Trainer(
                env=self.env, 
                eval_env=self.eval_env, 
                agent=self, 
                seed=self.seed, 
                )
            
            self.trainer.train(
                project_name=self.project_name, 
                load_path=self.load_path, 
                normalized_env=self.normalized_env,
                max_iters=self.max_iters, 
                eval_mode=self.eval_mode, 
                eval_intervals=self.eval_intervals, 
                eval_iters=self.eval_iters, 
                show_stats=self.show_stats, 
                show_graphs=self.show_graphs,
                )
            
    def step(self, state, action, reward, next_state, done):
        result = None
        self.buffer.store(state, action, reward, next_state, done)
        if self.prioritized_mode:
            fraction = min(self.timesteps / self.max_iters, 1.)
            self.prio_beta = self.prio_beta + fraction * (1. - self.prio_beta)

            if self.buffer.size >= self.update_after:
                states, actions, rewards, next_states, dones, weights, idxs = self.buffer.sample(self.prio_beta)
                result = self.learn(states, actions, rewards, next_states, dones, weights)

                if result['td_error'] is not None:
                    td_error = result['td_error'].detach().cpu().abs().numpy().flatten()
                    new_prios = td_error + self.prio_eps
                    self.buffer.update_priorities(idxs, new_prios)
        else:
            if self.buffer.size >= self.update_after:
                states, actions, rewards, next_states, dones = self.buffer.sample()
                result = self.learn(states, actions, rewards, next_states, dones)

        return result

    def soft_update(self, main_model, target_model):
        for target_param, main_param in zip(target_model.parameters(), main_model.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1. - self.tau) * target_param.data)

    def save(self, *args, **kwargs):
        raise NotImplementedError()

    def load(self, *args, **kwargs):
        raise NotImplementedError()
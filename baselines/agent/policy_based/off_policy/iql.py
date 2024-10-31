import torch, torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam
from baselines.common.policy import OffPolicyAlgorithm
from baselines.common.network import MLPGaussianPolicy, MLPDeterministicPolicy, MLPVFunction, MLPDoubleQFunction
from baselines.common.noise import GaussianNoise, OrnsteinUhlenbeckNoise


class IQL(OffPolicyAlgorithm):
    def __init__(self, env, **config):
        super().__init__(
            env=env,
            actor_size=config.get('actor_size', (256, 256)),
            critic_size=config.get('critic_size', (256, 256)),
            actor_activation=config.get('actor_activation', torch.relu),
            critic_activation=config.get('critic_activation', torch.relu),
            buffer_size=config.get('buffer_size', int(1e+6)),
            batch_size=config.get('batch_size', 256),
            update_after=config.get('update_after', 1000),
            actor_lr=config.get('actor_lr', 3e-4),
            critic_lr=config.get('critic_lr', 3e-4),
            gamma=config.get('gamma', 0.99),
            tau=config.get('tau', 0.005),
            reward_norm=config.get('reward_norm', False),
            prioritized_mode=config.get('prioritized_mode', False),
            prio_alpha=config.get('prio_alpha', 0.6),
            prio_beta=config.get('prio_beta', 0.4),
            prio_eps=config.get('prio_eps', 1e-6)
        )

        self.value_size = config.get('value_size', (256, 256))
        self.value_activation = config.get('value_activation', torch.relu)
        self.value_lr = config.get('value_lr', 3e-4)
        self.max_weight = config.get('max_weight', 100.0)
        self.expectile = config.get('expectile', 0.7)
        self.temperature = config.get('temperature', 3.0)
        self.policy_mode = config.get('policy_mode', 'gaussian')
        self.noise_type = config.get('noise_type', 'normal')
        self.action_noise_std = config.get('action_noise_std', 0.1)
        self.config = config
        
        if self.policy_mode == 'deterministic':
            self.actor = MLPDeterministicPolicy(
                self.state_dim, 
                self.action_dim, 
                self.actor_size, 
                self.actor_activation
                ).to(self.device)
        else:
            self.actor = MLPGaussianPolicy(
                self.state_dim, 
                self.action_dim, 
                self.actor_size, 
                self.actor_activation
                ).to(self.device)

        self.critic = MLPDoubleQFunction(
            self.state_dim, 
            self.action_dim,
            self.critic_size, 
            self.critic_activation
            ).to(self.device)
        self.target_critic = deepcopy(self.critic)

        self.value = MLPVFunction(
            self.state_dim, 
            self.value_size, 
            self.value_activation
            ).to(self.device)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)
        self.value_optim = Adam(self.value.parameters(), lr=self.value_lr)

        if self.noise_type == 'normal':
            self.noise = GaussianNoise(
                self.action_dim, 
                sigma=self.action_noise_std, 
                device=self.device
                )
        elif self.noise_type == 'ou':
            self.noise = OrnsteinUhlenbeckNoise(
                self.action_dim, 
                sigma=self.action_noise_std, 
                device=self.device
                )
    
    @torch.no_grad()
    def act(self, state, training=True):
        if (self.buffer.ptr < self.update_after) and training:
            self.random_action()

        self.actor.train(training)
        state = torch.FloatTensor(state).to(self.device)
        if self.policy_mode == 'deterministic':
            action = self.actor(state)
            action = torch.clamp(action + self.noise.sample(), -1., 1.) if training else action
            return action.cpu().numpy()        
        else:
            mu, std = self.actor(state)
            action = torch.normal(mu, std) if training else mu
            return torch.tanh(action).cpu().numpy()
    
    def learn(self, states, actions, rewards, next_states, dones, weights=None):
        self.actor.train()
        self.critic.train()
        self.value.train()

        with torch.no_grad():
            values, next_values = self.value(states), self.value(next_states)
            target_q1_values, target_q2_values = self.target_critic(states, actions)
            value_target_values = torch.min(target_q1_values, target_q2_values)
            critic_target_values = rewards + (1. - dones) * self.gamma * next_values

            actor_td_error = value_target_values - values
            exp_a = torch.exp(actor_td_error * self.temperature)
            exp_a = torch.clamp(exp_a, max=self.max_weight)

        value_td_error = value_target_values - self.value(states)
        expectile_weight = torch.where(value_td_error > 0, self.expectile, 1 - self.expectile)
        if self.prioritized_mode:
            value_loss = (weights * expectile_weight * (value_td_error ** 2)).mean()
        else:        
            value_loss = (expectile_weight * (value_td_error ** 2)).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        
        if self.policy_mode == 'deterministic':
            pi = self.actor(states)
            if self.prioritized_mode:
                actor_loss = (weights * exp_a * (pi - actions) ** 2).mean()
            else:
                actor_loss = (exp_a * (pi - actions) ** 2).mean()
        else:
            log_probs = self.actor.log_prob(states, actions)
            if self.prioritized_mode:
                actor_loss = -(weights * exp_a * log_probs).mean()
            else:
                actor_loss = -(exp_a * log_probs).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        q1_values, q2_values = self.critic(states, actions)
        if self.prioritized_mode:
            td_error1 = critic_target_values - q1_values
            td_error2 = critic_target_values - q2_values
            td_error = td_error1 + td_error2 
            critic_loss = (weights * td_error1 ** 2).mean() + (weights * td_error2 ** 2).mean()
        else:
            critic_loss = F.mse_loss(q1_values, critic_target_values) + F.mse_loss(q2_values, critic_target_values)
            
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        entropy = torch.tensor(0.) if self.policy_mode == 'deterministic' else self.actor.entropy(states)
        self.soft_update(self.critic, self.target_critic)
        
        result = {
            'agent_timesteps': self.timesteps, 
            'actor_loss': actor_loss.item(), 
            'critic_loss': critic_loss.item(), 
            'value_loss': value_loss.item(), 
            'entropy': entropy.item(),
            'td_error': td_error
            }
                
        return result
        
    def save(self, save_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
            'value_optim_state_dict': self.value_optim.state_dict()
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.value_optim.load_state_dict(checkpoint['value_optim_state_dict'])
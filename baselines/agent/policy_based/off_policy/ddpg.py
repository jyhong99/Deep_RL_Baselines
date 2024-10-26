import torch, torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
from baselines.common.policy import OffPolicyAlgorithm
from baselines.common.network import MLPDeterministicPolicy, MLPQFunction
from baselines.common.noise import GaussianNoise, OrnsteinUhlenbeckNoise


class DDPG(OffPolicyAlgorithm):
    def __init__(self, env, **config):
        super().__init__(
            env=env,
            actor_size=config.get('actor_sizes', (256, 256)),
            critic_size=config.get('critic_sizes', (256, 256)),
            actor_activation=config.get('actor_activation', torch.relu),
            critic_activation=config.get('critic_activation', torch.relu),
            buffer_size=config.get('buffer_size', int(1e+6)),
            batch_size=config.get('batch_size', 256),
            update_after=config.get('update_after', 1000),
            actor_lr=config.get('actor_lr', 1e-3),
            critic_lr=config.get('critic_lr', 1e-3),
            gamma=config.get('gamma', 0.99),
            tau=config.get('tau', 0.005),
            reward_norm=config.get('reward_norm', False),
            prioritized_mode=config.get('prioritized_mode', False),
            prio_alpha=config.get('prio_alpha', 0.6),
            prio_beta=config.get('prio_beta', 0.4),
            prio_eps=config.get('prio_eps', 1e-6)
        )

        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.action_noise_std = config.get('action_noise_std', 0.1)
        self.noise_type = config.get('noise_type', 'normal')
        self.config = config

        self.actor = MLPDeterministicPolicy(
            self.state_dim, 
            self.action_dim, 
            self.actor_size, 
            self.actor_activation
            ).to(self.device)
        self.target_actor = deepcopy(self.actor)

        self.critic = MLPQFunction(
            self.state_dim, 
            self.action_dim, 
            self.critic_size, 
            self.critic_activation
            ).to(self.device)
        self.target_critic = deepcopy(self.critic)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

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
        if (self.buffer.size < self.update_after) and training:
            self.random_action()

        self.actor.train(training)
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        if training:
            action += self.noise.sample()

        return torch.clamp(action, -1., 1.).cpu().numpy()

    def learn(self, states, actions, rewards, next_states, dones, weights=None):
        self.actor.train()
        self.critic.train()

        with torch.no_grad():
            next_q_values = self.target_critic(next_states, self.target_actor(next_states))
            target_q_values = rewards + (1. - dones) * self.gamma * next_q_values

        td_error = None
        q_values = self.critic(states, actions)
        if self.prioritized_mode:
            td_error = target_q_values - q_values
            critic_loss = (weights * td_error ** 2).mean()
        else:
            critic_loss = F.mse_loss(q_values, target_q_values)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        result = {
            'agent_timesteps': self.timesteps, 
            'actor_loss': actor_loss.item(), 
            'critic_loss': critic_loss.item(), 
            'td_error': td_error
            }
    
        return result

    def save(self, save_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict()
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
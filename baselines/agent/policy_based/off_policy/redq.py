import random, numpy as np
import torch, torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam
from baselines.common.policy import OffPolicyAlgorithm
from baselines.common.network import MLPGaussianPolicy, MLPGaussianSDEPolicy, MLPQFunction


class REDQ(OffPolicyAlgorithm):
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

        self.update_freq = config.get('update_freq', 1)
        self.alpha = config.get('alpha', 0.2)
        self.n_critics = config.get('n_critics', 10)
        self.n_critic_samples = config.get('n_critic_samples', 2)
        self.vf_iters = config.get('vf_iters', 20)
        self.gsde_mode = config.get('gsde_mode', False)
        self.adaptive_alpha_mode = config.get('adaptive_alpha_mode', False)
        self.ent_lr = config.get('ent_lr', 3e-4)
        self.config = config

        if self.gsde_mode:
            self.actor = MLPGaussianSDEPolicy(
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
        
        self.critics = [
            MLPQFunction(
                self.state_dim, 
                self.action_dim, 
                self.critic_size, 
                self.critic_activation
                ).to(self.device) 
                for _ in range(self.n_critics)
            ]
        
        self.target_critics = [
            deepcopy(critic) 
            for critic in self.critics
            ]

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optims = [Adam(critic.parameters(), lr=self.critic_lr) for critic in self.critics]

        if self.adaptive_alpha_mode:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.ent_lr)

    @torch.no_grad()
    def act(self, state, training=True, global_buffer_size=None):
        if global_buffer_size is None:
            if (self.buffer.size < self.update_after) and training:
                return self.random_action()
        else:
            if (global_buffer_size < self.update_after) and training:
                return self.random_action()
        
        self.actor.train(training)
        state = torch.FloatTensor(state).to(self.device)
        mu, std = self.actor(state)
        if self.gsde_mode:
            dist = self.actor.dist(state)
            action = dist.sample() if training else mu
            return torch.tanh(action + self.actor.get_noise()).cpu().numpy()
        else:
            action = torch.normal(mu, std) if training else mu
            return torch.tanh(action).cpu().numpy()
    
    def learn(self, 
              states, actions, rewards, next_states, dones, 
              weights=None, global_timesteps=None
            ):
        
        self.actor.train()
        for critic in self.critics:
            critic.train()

        if global_timesteps is not None:
            self.timesteps = global_timesteps

        if self.gsde_mode:
            self.actor.reset_noise()
        
        total_td_error, critic_losses = 0, []
        idxs = random.sample(range(self.n_critics), self.n_critic_samples)
        for _ in range(self.vf_iters):
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states)
                next_q_values = torch.stack([self.target_critics[i](next_states, next_actions) for i in idxs])
                next_mean_q_values = torch.mean(next_q_values, dim=0)
                target_q_values = rewards + (1. - dones) * self.gamma * (next_mean_q_values - self.alpha * next_log_probs)

            q_values = [critic(states, actions) for critic in self.critics]
            for q_value, critic_optim in zip(q_values, self.critic_optims):
                if self.prioritized_mode:
                    td_error = target_q_values - q_value
                    total_td_error += td_error
                    critic_loss = (weights * td_error ** 2).mean()
                else:
                    critic_loss = F.mse_loss(q_value, target_q_values)
                critic_losses.append(critic_loss.item())

                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

            for critic, target_critic in zip(self.critics, self.target_critics):
                self.soft_update(critic, target_critic)

        sample_actions, log_probs = self.actor.sample(states)
        q_values = torch.stack([critic(states, sample_actions) for critic in self.critics])
        mean_q_values = torch.mean(q_values, dim=0)
        if self.prioritized_mode:
            actor_loss = -(weights * (mean_q_values - self.alpha * log_probs)).mean()
        else:
            actor_loss = -(mean_q_values - self.alpha * log_probs).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = torch.tensor(0.)
        if self.adaptive_alpha_mode:
            if self.prioritized_mode:
                alpha_loss = -(weights * self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            else:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        entropy = self.actor.entropy(states)
        result = {
        'agent_timesteps': self.timesteps, 
        'actor_loss': actor_loss.item(), 
        'critic_loss': np.mean(critic_losses), 
        'alpha_loss': alpha_loss.item(),
        'entropy': entropy.item(),
        'alpha': self.alpha,
        'td_error': total_td_error / self.vf_iters
        }
                
        return result
    
    def save(self, save_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critics_state_dict': [critic.state_dict() for critic in self.critics],
            'target_critics_state_dict': [target_critic.state_dict() for target_critic in self.target_critics],
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optims_state_dict': [critic_optim.state_dict() for critic_optim in self.critic_optims]
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])

        for critic, state_dict in zip(self.critics, checkpoint['critics_state_dict']):
            critic.load_state_dict(state_dict)

        for target_critic, state_dict in zip(self.target_critics, checkpoint['target_critics_state_dict']):
            target_critic.load_state_dict(state_dict)
        
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        for critic_optim, state_dict in zip(self.critic_optims, checkpoint['critic_optims_state_dict']):
            critic_optim.load_state_dict(state_dict)
import torch
from copy import deepcopy
from torch.optim import Adam
from baselines.common.policy import OffPolicyAlgorithm
from baselines.common.network import MLPGaussianPolicy, MLPGaussianSDEPolicy, MLPQuantileQFunction
from baselines.common.operation import quantile_huber_loss


class TQC(OffPolicyAlgorithm):
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

        self.n_quantiles = config.get('n_quantiles', 25)
        self.n_networks = config.get('n_networks', 5)
        self.top_quantiles_to_drop_per_net = config.get('top_quantiles_to_drop_per_net', 2)
        self.alpha = config.get('alpha', 0.2)
        self.gsde_mode = config.get('gsde_mode', False)
        self.adaptive_alpha_mode = config.get('adaptive_alpha_mode', False)
        self.ent_lr = config.get('ent_lr', 3e-4)
        self.config = config

        self.top_quantiles_to_drop = self.top_quantiles_to_drop_per_net * self.n_networks
        self.quantiles_total = self.n_quantiles * self.n_networks

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
            
        
        self.critic = MLPQuantileQFunction(
            self.state_dim, 
            self.action_dim, 
            self.n_quantiles, 
            self.n_networks, 
            self.critic_size, 
            self.critic_activation
            ).to(self.device)
        self.target_critic = deepcopy(self.critic)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

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
        self.critic.train()
        if global_timesteps is not None:
            self.timesteps = global_timesteps
            
        if self.gsde_mode:
            self.actor.reset_noise()
            
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_quantiles = self.target_critic(next_states, next_actions)
            n_target_quantiles = self.quantiles_total - self.top_quantiles_to_drop
            next_quantiles = torch.sort(next_quantiles.reshape(self.batch_size, -1))[0]
            next_quantiles = next_quantiles[:, :n_target_quantiles]
            target_quantiles = rewards + (1. - dones) * self.gamma * (next_quantiles - self.alpha * next_log_probs.reshape(-1, 1))
            target_quantiles = target_quantiles.unsqueeze_(dim=1)
        
        curr_quantiles = self.critic(states, actions)
        critic_loss, td_error = quantile_huber_loss(curr_quantiles, target_quantiles, self.device, weights, sum_over_quantiles=False)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        sample_actions, log_probs = self.actor.sample(states)
        sample_quantiles = self.critic(states, sample_actions).mean(dim=2).mean(dim=1, keepdim=True)

        if self.prioritized_mode:
            actor_loss = -(weights * (sample_quantiles - self.alpha * log_probs.reshape(-1, 1))).mean()
        else:
            actor_loss = -(sample_quantiles - self.alpha * log_probs.reshape(-1, 1)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = torch.tensor(0.)        
        if self.adaptive_alpha_mode:
            if self.prioritized_mode:
                alpha_loss = -(weights * (self.log_alpha * (log_probs + self.target_entropy).detach())).mean()
            else:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

            self.soft_update(self.critic, self.target_critic)

        entropy = self.actor.entropy(states)
        result = {
            'agent_timesteps': self.timesteps, 
            'actor_loss': actor_loss.item(), 
            'critic_loss': critic_loss.item(), 
            'alpha_loss': alpha_loss.item(),
            'entropy': entropy.item(),
            'alpha': self.alpha,
            'td_error': td_error
            }
                
        return result

    def save(self, save_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict()
        }, save_path)
        
        if self.adaptive_alpha_mode:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'target_critic_state_dict': self.target_critic.state_dict(),
                'actor_optim_state_dict': self.actor_optim.state_dict(),
                'critic_optim_state_dict': self.critic_optim.state_dict(),
                'alpha_optim_state_dict': self.alpha_optim.state_dict()
            }, save_path)            

    def load(self, load_path):
        checkpoint = torch.load(load_path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])

        if self.adaptive_alpha_mode:
            self.alpha_optim.load_state_dict(checkpoint['alpha_optim_state_dict'])
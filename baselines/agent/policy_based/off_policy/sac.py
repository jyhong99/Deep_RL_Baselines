import torch, torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam
from baselines.common.policy import OffPolicyAlgorithm
from baselines.common.network import *


class SAC(OffPolicyAlgorithm):
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
            
        self.critic = MLPDoubleQFunction(
            self.state_dim, 
            self.action_dim, 
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
    def act(self, state, global_buffer_size=None, training=True):
        self.timesteps += 1

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
    
    def learn(self, states, actions, rewards, next_states, dones, weights=None):
        self.actor.train()
        self.critic.train()

        if self.gsde_mode:
            self.actor.reset_noise()

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1_values, next_q2_values = self.target_critic(next_states, next_actions)
            next_q_values = torch.min(next_q1_values, next_q2_values)
            target_q_values = rewards + (1. - dones) * self.gamma * (next_q_values - self.alpha * next_log_probs)

        td_error = None
        q1_values, q2_values = self.critic(states, actions)

        if self.prioritized_mode:
            td_error1 = target_q_values - q1_values
            td_error2 = target_q_values - q2_values
            td_error = td_error1 + td_error2 
            critic_loss = (weights * td_error1 ** 2).mean() + (weights * td_error2 ** 2).mean()
        else:
            critic_loss = F.mse_loss(q1_values, target_q_values) + F.mse_loss(q2_values, target_q_values)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = torch.tensor(0.)
        if self.timesteps % self.update_freq == 0:
            sample_actions, log_probs = self.actor.sample(states)
            q1_values, q2_values = self.critic(states, sample_actions)
            q_val = torch.min(q1_values, q2_values)

            if self.prioritized_mode:
                actor_loss = -(weights * (q_val - self.alpha * log_probs)).mean()
            else:
                actor_loss = -(q_val - self.alpha * log_probs).mean()            

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

    def load(self, load_path):
        checkpoint = torch.load(load_path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])




class SAC_Discrete(OffPolicyAlgorithm):
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
        self.dueling_mode = config.get('gsde_mode', False)
        self.gsde_mode = config.get('gsde_mode', False)
        self.adaptive_alpha_mode = config.get('adaptive_alpha_mode', False)
        self.ent_lr = config.get('ent_lr', 3e-4)
        self.config = config

        if self.action_type == 'discrete':
            self.actor = MLPCategoricalPolicy(
                self.state_dim, 
                self.action_dim, 
                self.actor_size, 
                self.actor_activation
            ).to(self.device)

            self.critic = MLPDoubleQNetwork(
                self.state_dim, 
                self.action_dim, 
                self.critic_size, 
                self.critic_activation,
                self.dueling_mode
            ).to(self.device)

        elif self.action_type == 'multidiscrete':
            self.actor = MLPMultiCategoricalPolicy(
                self.state_dim, 
                self.action_dim, 
                self.actor_size, 
                self.actor_activation
             ).to(self.device)
            
            self.critic = MLPMultiDoubleQNetwork(
                self.state_dim, 
                self.action_dim, 
                self.critic_size, 
                self.critic_activation,
                self.dueling_mode
            ).to(self.device)

        self.target_critic = deepcopy(self.critic)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        self.adaptive_alpha_mode = self.adaptive_alpha_mode
        if self.adaptive_alpha_mode:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.ent_lr)

    @torch.no_grad()
    def act(self, state, global_buffer_size=None, training=True):
        self.timesteps += 1

        if global_buffer_size is None:
            if (self.buffer.size < self.update_after) and training:
                return self.random_action()
        else:
            if (global_buffer_size < self.update_after) and training:
                return self.random_action()
        
        self.actor.train(training)
        state = torch.FloatTensor(state).to(self.device)
        logits = self.actor(state)
        if self.action_type == 'discrete':
            if training:
                action = torch.multinomial(logits, 1)
            else:
                action = torch.argmax(logits, dim=-1, keepdim=True)
            return action.item()
        
        elif self.action_type == 'multidiscrete':
            if training:
                actions = [torch.multinomial(logit, 1).item() for logit in logits] 
            else:
                actions = [torch.argmax(logit).item() for logit in logits]
            return actions
    
    def learn(self, states, actions, rewards, next_states, dones, weights=None):
        self.actor.train()
        self.critic.train()

        with torch.no_grad():
            next_probs = self.actor.sample(next_states)
            next_log_probs = torch.log(next_probs)
            next_q1_values, next_q2_values = self.target_critic(next_states)
            next_q_values = torch.min(next_q1_values, next_q2_values)
            next_values = (next_probs * (next_q_values - self.alpha * next_log_probs)).sum(-1).unsqueeze(-1)
            target_q_values = rewards + (1. - dones) * self.gamma * next_values

        td_error = None
        q1_values, q2_values = self.critic(states)
        q1_values = q1_values.gather(1, actions.long())
        q2_values = q2_values.gather(1, actions.long())

        if self.prioritized_mode:
            td_error1 = target_q_values - q1_values
            td_error2 = target_q_values - q2_values
            td_error = td_error1 + td_error2 
            critic_loss = (weights * td_error1 ** 2).mean() + (weights * td_error2 ** 2).mean()
        else:
            critic_loss = F.mse_loss(q1_values, target_q_values) + F.mse_loss(q2_values, target_q_values)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = torch.tensor(0.)
        if self.timesteps % self.update_freq == 0:
            probs = self.actor.sample(states)
            log_probs = torch.log(probs)

            with torch.no_grad():
                q1_values, q2_values = self.critic(states)
                q_values = torch.min(q1_values, q2_values)
                
            if self.prioritized_mode:
                actor_loss = -(weights * probs * (q_values - self.alpha * log_probs)).sum(-1).mean()
            else:
                actor_loss = -(probs * (q_values - self.alpha * log_probs)).sum(-1).mean()            

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            alpha_loss = torch.tensor(0.)
            if self.adaptive_alpha_mode:
                log_probs = (probs * log_probs).sum(-1)

                if self.prioritized_mode:
                    alpha_loss = -(weights * self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
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

    def load(self, load_path):
        checkpoint = torch.load(load_path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
import numpy as np
import torch, torch.nn.functional as F
from torch.optim import Adam
from baselines.common.policy import OnPolicyAlgorithm
from torch.utils.data import TensorDataset, DataLoader
from baselines.common.network import *


class PPO(OnPolicyAlgorithm):
    def __init__(self, env, **config):
        super().__init__(
            env=env,
            actor_size=config.get('actor_size', (64, 64)),
            critic_size=config.get('critic_size', (64, 64)),
            actor_activation=config.get('actor_activation', torch.tanh),
            critic_activation=config.get('critic_activation', torch.tanh),
            buffer_size=config.get('buffer_size', int(1e+6)),
            update_after=config.get('update_after', 2048),
            actor_lr=config.get('step_size', 3e-4),
            critic_lr=None,
            gamma=config.get('gamma', 0.99),
            lmda=config.get('lmda', 0.95),
            vf_coef=config.get('vf_coef', 1.0),  
            ent_coef=config.get('ent_coef', 0.01),        
            reward_norm=config.get('reward_norm', False),
            adv_norm=config.get('adv_norm', False)
        )

        self.train_iters = config.get('train_iters', 10)
        self.batch_size = config.get('batch_size', 64)
        self.clip_range = config.get('clip_range', 0.2)
        self.clip_range_vf = config.get('clip_range_vf', None)
        self.target_kl = config.get('target_kl', None)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.gsde_mode = config.get('gsde_mode', False)
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
        
        self.critic = MLPVFunction(
            self.state_dim, 
            self.critic_size, 
            self.critic_activation
            ).to(self.device)

        self.optim = Adam(self.actor.parameters(), lr=self.actor_lr)

    @torch.no_grad()
    def act(self, state, training=True, global_buffer_size=None):
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
                
    def learn(self, states, actions, rewards, next_states, dones):
        self.actor.train()
        self.critic.train()

        if self.gsde_mode:
            self.actor.reset_noise()
                        
        with torch.no_grad():
            values, next_values = self.critic(states), self.critic(next_states)
            rets, advs = self.GAE(values, next_values, rewards, dones)
            log_prob_olds = self.actor.log_prob(states, actions)
            
        continue_training = True
        actor_losses, critic_losses, entropies = [], [], []
        losses, clip_fracs, approx_kls = [], [], []
        
        dataset = TensorDataset(states, actions, values, rets, advs, log_prob_olds)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.train_iters):
            for batch in dataloader:
                batch_states, batch_actions, batch_values, batch_rets, batch_advs, batch_log_prob_olds = batch

                log_probs = self.actor.log_prob(batch_states, batch_actions)
                ratios = (log_probs - batch_log_prob_olds).exp()
                surr1 = batch_advs * ratios
                surr2 = batch_advs * torch.clamp(ratios, 1. - self.clip_range, 1. + self.clip_range)
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_losses.append(actor_loss.item())

                clip_frac = torch.mean((torch.abs(ratios - 1) > self.clip_range).float())
                clip_fracs.append(clip_frac.item())

                values = self.critic(batch_states)
                if self.clip_range_vf is not None:
                    clipped_values = batch_values + torch.clamp(values - batch_values, -self.clip_range_vf, self.clip_range_vf)
                    critic_loss1 = F.mse_loss(values, batch_rets)
                    critic_loss2 = F.mse_loss(clipped_values, batch_rets)
                    critic_loss = torch.max(critic_loss1, critic_loss2)
                else:
                    critic_loss = F.mse_loss(values, batch_rets)
                critic_losses.append(critic_loss.item())

                entropy = self.actor.entropy(batch_states)
                entropies.append(entropy.item())

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                losses.append(loss.item())

                with torch.no_grad():
                    log_ratios = log_probs - batch_log_prob_olds
                    approx_kl = torch.mean((torch.exp(log_ratios) - 1.) - log_ratios).cpu().numpy()
                    approx_kls.append(approx_kl)

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.optim.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optim.step()

            if not continue_training:
                break

        result = {
            'agent_timesteps': self.timesteps, 
            'actor_loss': np.mean(actor_losses), 
            'critic_loss': np.mean(critic_losses), 
            'total_loss': np.mean(losses), 
            'entropy': np.mean(entropies),
            'clip_frac': np.mean(clip_fracs), 
            'approx_kl': np.mean(approx_kls)       
            }
                
        return result

    def save(self, save_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])




class PPO_Discrete(OnPolicyAlgorithm):
    def __init__(self, env, **config):
        super().__init__(
            env=env,
            actor_size=config.get('actor_size', (64, 64)),
            critic_size=config.get('critic_size', (64, 64)),
            actor_activation=config.get('actor_activation', torch.tanh),
            critic_activation=config.get('critic_activation', torch.tanh),
            buffer_size=config.get('buffer_size', int(1e+6)),
            update_after=config.get('update_after', 2048),
            actor_lr=config.get('step_size', 3e-4),
            critic_lr=None,
            gamma=config.get('gamma', 0.99),
            lmda=config.get('lmda', 0.95),
            vf_coef=config.get('vf_coef', 1.0),  
            ent_coef=config.get('ent_coef', 0.01),        
            reward_norm=config.get('reward_norm', False),
            adv_norm=config.get('adv_norm', False),
        )

        self.train_iters = config.get('train_iters', 10)
        self.batch_size = config.get('batch_size', 64)
        self.clip_range = config.get('clip_range', 0.2)
        self.clip_range_vf = config.get('clip_range_vf', None)
        self.target_kl = config.get('target_kl', None)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.config = config
        
        if self.action_type == 'discrete':
            self.actor = MLPCategoricalPolicy(
                self.state_dim, 
                self.action_dim, 
                self.actor_size, 
                self.actor_activation
            ).to(self.device)
            
        elif self.action_type == 'multidiscrete':
            self.actor = MLPMultiCategoricalPolicy(
                self.state_dim, 
                self.action_dim, 
                self.actor_size, 
                self.actor_activation
             ).to(self.device)
        
        self.critic = MLPVFunction(
            self.state_dim, 
            self.critic_size, 
            self.critic_activation
            ).to(self.device)

        self.optim = Adam(self.actor.parameters(), lr=self.actor_lr)

    @torch.no_grad()
    def act(self, state, training=True, global_buffer_size=None):
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
        
    def learn(self, states, actions, rewards, next_states, dones):
        self.actor.train()
        self.critic.train()
                        
        with torch.no_grad():
            values, next_values = self.critic(states), self.critic(next_states)
            rets, advs = self.GAE(values, next_values, rewards, dones)
            log_prob_olds = self.actor.log_prob(states, actions)
            
        continue_training = True
        actor_losses, critic_losses, entropies = [], [], []
        losses, clip_fracs, approx_kls = [], [], []
        
        dataset = TensorDataset(states, actions, values, rets, advs, log_prob_olds)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.train_iters):
            for batch in dataloader:
                batch_states, batch_actions, batch_values, batch_rets, batch_advs, batch_log_prob_olds = batch

                log_probs = self.actor.log_prob(batch_states, batch_actions)
                ratios = (log_probs - batch_log_prob_olds).exp()
                surr1 = batch_advs * ratios
                surr2 = batch_advs * torch.clamp(ratios, 1.-self.clip_range, 1.+self.clip_range)
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_losses.append(actor_loss.item())

                clip_frac = torch.mean((torch.abs(ratios - 1.) > self.clip_range).float())
                clip_fracs.append(clip_frac.item())

                values = self.critic(batch_states)
                if self.clip_range_vf is not None:
                    clipped_value = batch_values + torch.clamp(values - batch_values, -self.clip_range_vf, self.clip_range_vf)
                    critic_loss1 = F.mse_loss(values, batch_rets)
                    critic_loss2 = F.mse_loss(clipped_value, batch_rets)
                    critic_loss = torch.max(critic_loss1, critic_loss2)
                else:
                    critic_loss = F.mse_loss(values, batch_rets)
                critic_losses.append(critic_loss.item())

                entropy = self.actor.entropy(batch_states)
                entropies.append(entropy.item())

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                losses.append(loss.item())

                with torch.no_grad():
                    log_ratio = log_probs - batch_log_prob_olds
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1.) - log_ratio).cpu().numpy()
                    approx_kls.append(approx_kl)

                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.optim.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optim.step()

            if not continue_training:
                break

        result = {
            'agent_timesteps': self.timesteps, 
            'actor_loss': np.mean(actor_losses), 
            'critic_loss': np.mean(critic_losses), 
            'total_loss': np.mean(losses), 
            'entropy': np.mean(entropies),
            'clip_frac': np.mean(clip_fracs), 
            'approx_kl': np.mean(approx_kls)       
            }
                
        return result

    def save(self, save_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
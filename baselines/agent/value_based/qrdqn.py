import random
import torch
from copy import deepcopy
from torch.optim import Adam
from baselines.common.policy import OffPolicyAlgorithm
from baselines.common.network import MLPQuantileQNetwork
from baselines.common.operation import quantile_huber_loss


class QRDQN(OffPolicyAlgorithm):
    def __init__(self, env, **config):
        super().__init__(
            env=env,
            actor_size=config.get('policy_size', (256, 256)),
            critic_size=None,
            actor_activation=config.get('policy_activation', torch.relu),
            critic_activation=None,
            buffer_size=config.get('buffer_size', int(1e+6)),
            batch_size=config.get('batch_size', 256),
            update_after=config.get('update_after', 1000),
            actor_lr=config.get('policy_lr', 3e-4),
            critic_lr=None,
            gamma=config.get('gamma', 0.99),
            tau=config.get('tau', 0.005),
            reward_norm=config.get('reward_norm', False),
            prioritized_mode=config.get('prioritized_mode', False),
            prio_alpha=config.get('prio_alpha', 0.6),
            prio_beta=config.get('prio_beta', 0.4),
            prio_eps=config.get('prio_eps', 1e-6)
        )
        
        self.n_quantiles = config.get('n_quantiles', 200)
        self.eps_start = config.get('eps_start', 1.0)
        self.eps_min = config.get('eps_min', 0.01)
        self.eps_decay = config.get('eps_decay', 0.999)
        self.eps_threshold = self.eps_start
        self.max_grad_norm = config.get('max_grad_norm', 10)
        self.double_mode = config.get('double_mode', False)
        self.dueling_mode = config.get('dueling_mode', False)
        self.config = config

        self.policy = MLPQuantileQNetwork(
            self.state_dim, 
            self.action_dim, 
            self.n_quantiles,
            self.actor_size, 
            self.actor_activation,
            self.dueling_mode
            ).to(self.device)
        self.target_policy = deepcopy(self.policy)
        self.optim = Adam(self.policy.parameters(), lr=self.actor_lr)

    @torch.no_grad()
    def act(self, state, training=True, global_buffer_size=None):
        if global_buffer_size is None:
            if (self.buffer.size < self.update_after) and training:
                return self.random_action()
        else:
            if (global_buffer_size < self.update_after) and training:
                return self.random_action()

        if self.eps_threshold > self.eps_min:
            self.eps_threshold *= self.eps_decay

        self.policy.train(training)
        if random.random() <= self.eps_threshold:
            return self.random_action()
        
        state = torch.FloatTensor(state).to(self.device)
        action = self.policy(state).mean(dim=1).argmax(dim=1).reshape(-1)
        return action.item()
    
    def learn(self, 
              states, actions, rewards, next_states, dones, 
              weights=None, global_timesteps=None
            ):
        
        self.policy.train()
        if global_timesteps is not None:
            self.timesteps = global_timesteps

        with torch.no_grad():
            next_quantiles = self.policy(next_states) if self.double_mode else self.target_policy(next_states)
            next_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
            next_actions = next_actions.expand(self.batch_size, self.n_quantiles, 1)
            next_quantiles = next_quantiles.gather(dim=2, index=next_actions).squeeze(dim=2)
            target_quantiles = rewards + (1. - dones) * self.gamma * next_quantiles

        curr_quantiles = self.policy(states)
        actions = actions[..., None][:, :1, :].long().expand(self.batch_size, self.n_quantiles, 1)
        curr_quantiles = curr_quantiles.gather(dim=2, index=actions).squeeze(dim=2)
        loss, td_error = quantile_huber_loss(curr_quantiles, target_quantiles, weights=weights, device=self.device)  

        self.optim.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optim.step()

        self.soft_update(self.policy, self.target_policy)

        result = {
            'agent_timesteps': self.timesteps, 
            'loss': loss.item(), 
            'td_error': td_error
            }
                
        return result

    def save(self, save_path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'target_policy_state_dict': self.target_policy.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path, weights_only=True)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.target_policy.load_state_dict(checkpoint['target_policy_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
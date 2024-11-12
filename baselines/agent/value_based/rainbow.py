import torch
from copy import deepcopy
from torch.optim import Adam
from baselines.common.policy import OffPolicyAlgorithm
from baselines.common.network import MLPRainbowQNetwork


class RainbowDQN(OffPolicyAlgorithm):
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
            prioritized_mode=True,
            prio_alpha=config.get('prio_alpha', 0.6),
            prio_beta=config.get('prio_beta', 0.4),
            prio_eps=config.get('prio_eps', 1e-6)
        )

        self.max_grad_norm = config.get('max_grad_norm', 10)
        self.v_min = config.get('v_min', 0.0)
        self.v_max = config.get('v_max', 200.0)
        self.atom_size = config.get('atom_size', 51)
        self.config = config

        self.support = torch.linspace(
            self.v_min, 
            self.v_max, 
            self.atom_size
            ).to(self.device)

        self.policy = MLPRainbowQNetwork(
            self.state_dim, 
            self.action_dim, 
            self.atom_size, 
            self.support, 
            self.actor_size, 
            self.actor_activation,
            ).to(self.device)
        self.target_policy = deepcopy(self.policy)
        self.optim = Adam(self.policy.parameters(), lr=self.actor_lr)

    @torch.no_grad()
    def act(self, state, training=True):
        self.timesteps += 1 
        if (self.buffer.size < self.update_after) and training:
            return self.random_action()
        
        self.policy.train(training)
        state = torch.FloatTensor(state).to(self.device)
        action = self.policy(state).argmax(dim=-1, keepdim=True)
        return action.item()

    def learn(self, states, actions, rewards, next_states, dones, weights):
        self.policy.train()

        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        with torch.no_grad():
            next_actions = self.policy(next_states).argmax(1)
            next_dist = self.target_policy.dist(next_states)
            next_dist = next_dist[range(self.batch_size), next_actions]

            t_z = rewards + (1. - dones) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size)
                .long().unsqueeze(1).expand(self.batch_size, self.atom_size).to(self.device)
                )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        dist = self.policy.dist(states)
        log_probs = torch.log(dist[range(self.batch_size), actions.long()[:, 0]])
        td_error = -(proj_dist * log_probs).sum(1)
        loss = (weights * td_error).mean()

        self.optim.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optim.step()

        self.policy.reset_noise()
        self.target_policy.reset_noise()
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
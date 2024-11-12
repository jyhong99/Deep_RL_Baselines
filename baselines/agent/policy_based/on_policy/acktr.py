import torch, torch.nn.functional as F
from baselines.common.optim import KFAC_optim
from baselines.common.policy import OnPolicyAlgorithm
from baselines.common.network import MLPGaussianPolicy, MLPGaussianSDEPolicy, MLPVFunction


class ACKTR(OnPolicyAlgorithm):
    def __init__(self, env, **config):
        super().__init__(
            env=env,
            actor_size=config.get('actor_size', (64, 64)),
            critic_size=config.get('critic_size', (64, 64)),
            actor_activation=config.get('actor_activation', torch.tanh),
            critic_activation=config.get('critic_activation', torch.tanh),
            buffer_size=config.get('buffer_size', int(1e+6)),
            update_after=config.get('update_after', 128),
            actor_lr=config.get('step_size', 3e-4),
            critic_lr=None,
            gamma=config.get('gamma', 0.99),
            lmda=config.get('lmda', 0.95),
            vf_coef=config.get('vf_coef', 1.0),  
            ent_coef=config.get('ent_coef', 0.01),        
            reward_norm=config.get('reward_norm', False),
            adv_norm=config.get('adv_norm', False)
        )

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
        
        self.optim = KFAC_optim(self.actor, lr=self.actor_lr)

    @torch.no_grad()
    def act(self, state, global_buffer_size=None, training=True):
        self.timesteps += 1
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
            advs = self.GAE(values, next_values, rewards, dones)[1]

        log_probs = self.actor.log_prob(states, actions)
        values = self.critic(states)
        sample_value = values + torch.randn(values.size()).to(self.device)
        
        actor_fisher_loss = -log_probs.mean()
        critic_fisher_loss = -F.mse_loss(values, sample_value.detach())
        fisher_loss = actor_fisher_loss + critic_fisher_loss

        self.optim.zero_grad()
        self.optim.fisher_backprop = True
        fisher_loss.backward(retain_graph=True)
        self.optim.fisher_backprop = False
        
        actor_loss = -(log_probs * advs).mean()
        critic_loss = advs.pow(2).mean()
        entropy = self.actor.entropy(states)
        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        result = {
            'agent_timesteps': self.timesteps, 
            'actor_loss': actor_loss.item(), 
            'critic_loss': critic_loss.item(), 
            'total_loss': loss.item(), 
            'actor_loss': actor_fisher_loss.item(), 
            'critic_loss': critic_fisher_loss.item(), 
            'total_loss': fisher_loss.item(), 
            'entropy': entropy.item()
            }
                
        return result
    
    def save(self, save_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
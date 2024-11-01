import numpy as np
import torch, torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam
from baselines.common.policy import OnPolicyAlgorithm
from baselines.common.network import MLPGaussianPolicy, MLPGaussianSDEPolicy, MLPVFunction


class TRPO(OnPolicyAlgorithm):
    def __init__(self, env, **config):
        super().__init__(
            env=env,
            actor_size=config.get('actor_size', (64, 64)),
            critic_size=config.get('critic_size', (64, 64)),
            actor_activation=config.get('actor_activation', torch.tanh),
            critic_activation=config.get('critic_activation', torch.tanh),
            buffer_size=config.get('buffer_size', int(1e+6)),
            update_after=config.get('update_after', 2048),
            actor_lr=None,
            critic_lr=config.get('vf_step_size', 3e-4),
            gamma=config.get('gamma', 0.99),
            lmda=config.get('lmda', 0.95),
            vf_coef=None,   
            ent_coef=None,          
            reward_norm=config.get('reward_norm', False),
            adv_norm=config.get('adv_norm', False)
        )

        self.actor_old = deepcopy(self.actor)
        self.delta = config.get('delta', 0.01)
        self.vf_iters = config.get('vf_iters', 10)
        self.backtrack_iters = config.get('backtrack_iters', 20)
        self.backtrack_coeff = config.get('backtrack_coeff', 1.0)
        self.backtrack_alpha = config.get('backtrack_alpha', 0.5)
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

        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

    @torch.no_grad()
    def act(self, state, training=True):
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
                       
    def _flat_grad(self, grads, hessian=False):
        grad_flatten = []
        if not hessian:
            for grad in grads:
                grad_flatten.append(grad.view(-1))
            grad_flatten = torch.cat(grad_flatten)
        else:
            for grad in grads:
                grad_flatten.append(grad.contiguous().view(-1))
            grad_flatten = torch.cat(grad_flatten).data
        return grad_flatten
    
    def _flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten
    
    def _update_model(self, model, new_params):
        index = 0
        for params in model.parameters():
            params_len = len(params.view(-1))
            new_param = new_params[index: index + params_len]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_len
    
    def _gaussian_kl(self, state, old_policy, policy):
        mu_old, std_old = old_policy(state)
        mu_old, std_old = mu_old.detach(), std_old.detach()
        mu, std = policy(state)
        kl = torch.log(std / std_old) + (std_old.pow(2) + (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
        return kl.sum(-1, keepdim=True).mean()
    
    def _fisher_vector_product(self, state, p, damping_coeff=0.1):
        p.detach()
        kl = self._gaussian_kl(state, self.actor_old, self.actor)
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad = self._flat_grad(kl_grad)

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian = torch.autograd.grad(kl_grad_p, self.actor.parameters())
        kl_hessian = self._flat_grad(kl_hessian, hessian=True)
        return kl_hessian + p * damping_coeff

    def _conjugate_gradient(self, state, b, cg_iters=10, eps=1e-8, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()

        rdotr = torch.dot(r, r)
        for _ in range(cg_iters):
            Ap = self._fisher_vector_product(state, p)
            alpha = rdotr / (torch.dot(p, Ap) + eps)

            x += alpha * p
            r -= alpha * Ap

            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p

            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def _compute_step_params(self, state, grad):
        search_dir = self._conjugate_gradient(state, grad.data)
        gHg = (self._fisher_vector_product(state, search_dir) * search_dir).sum(0)
        gHg = -gHg if gHg < 0 else gHg
        step_size = torch.sqrt(2 * self.delta / gHg)
        return search_dir, step_size

    def learn(self, states, actions, rewards, next_states, dones):
        self.actor.train()
        self.critic.train()

        if self.gsde_mode:
            self.actor.reset_noise()
            
        with torch.no_grad():
            values, next_values = self.critic(states), self.critic(next_states)
            rets, advs = self.GAE(values, next_values, rewards, dones)
            log_prob_olds = self.actor.log_prob(states, actions)

        critic_losses = []
        for _ in range(self.vf_iters):
            critic_loss = F.mse_loss(self.critic(states), rets)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            critic_losses.append(critic_loss.item())

        log_probs = self.actor.log_prob(states, actions)
        ratios = (log_probs - log_prob_olds).exp()
        actor_loss_old = (ratios * advs).mean()
        
        grad = torch.autograd.grad(actor_loss_old, self.actor.parameters())
        grad = self._flat_grad(grad)
        search_dir, step_size = self._compute_step_params(states, grad)
        old_params = self._flat_params(self.actor)
        self._update_model(self.actor_old, old_params)

        actor_losses = []
        with torch.no_grad():
            expected_improve = (grad * step_size * search_dir).sum(0, keepdim=True)
            for i in range(self.backtrack_iters):
                params = old_params + self.backtrack_coeff * step_size * search_dir
                self._update_model(self.actor, params)

                log_probs = self.actor.log_prob(states, actions)
                ratios = (log_probs - log_prob_olds).exp()
                actor_loss = (ratios * advs).mean()
                actor_losses.append(actor_loss.item())

                loss_improve = actor_loss - actor_loss_old
                expected_improve *= self.backtrack_coeff
                improve_condition = loss_improve / expected_improve

                kl = self._gaussian_kl(states, self.actor_old, self.actor)
                if kl < self.delta and improve_condition > self.backtrack_alpha:
                    break

                if i == self.backtrack_iters - 1:
                    params = self._flat_params(self.actor_old)
                    self._update_model(self.actor, params)
                self.backtrack_coeff *= 0.5
                
        entropy = self.actor.entropy(states)
        result = {
            'agent_timesteps': self.timesteps, 
            'actor_loss': np.mean(actor_losses), 
            'critic_loss': np.mean(critic_losses), 
            'entropy': entropy.item()
            }
                
        return result
    
    def save(self, save_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_old_state_dict': self.actor_old.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict()
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_old.load_state_dict(checkpoint['actor_old_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
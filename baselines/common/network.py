import math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal, Categorical
from baselines.common.operation import TanhBijector

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation):
        super(MLPBase, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.activation = activation
    
    def forward(self, state):
        x = self.activation(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        return x
    
    def mlp(self, state):
        x = self.activation(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        return x


# https://github.com/Kaixhin/Rainbow/blob/master/model.py
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class MLPNoisyBase(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation):
        super(MLPNoisyBase, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(NoisyLinear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.activation = activation
    
    def mlp(self, state):
        x = self.activation(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        return x
    
    def base_reset_noise(self):
        for hidden_layer in self.hidden_layers:
            hidden_layer.reset_noise()








class MLPDeterministicPolicy(MLPBase):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        super(MLPDeterministicPolicy, self).__init__(state_dim, hidden_sizes, activation)
        self.output_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = self.mlp(state)
        return torch.tanh(self.output_layer(x))


class MLPCategoricalPolicy(MLPBase):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU, epsilon=1e-8):
        super(MLPCategoricalPolicy, self).__init__(state_dim, hidden_sizes, activation)
        self.epsilon = epsilon
        self.output_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = self.mlp(state)
        logits = F.softmax(self.output_layer(x), dim=-1)
        return logits

    def dist(self, state):
        return Categorical(self.forward(state))

    def log_prob(self, state, action):
        dist = self.dist(state)
        return dist.log_prob(action)

    def entropy(self, state):
        dist = self.dist(state)
        return dist.entropy().mean()
    
    def sample(self, state):
        logits = self.forward(state)
        z = self.epsilon * (logits == 0.0).float()
        return logits + z
        

class MLPGaussianPolicy(MLPBase):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU, epsilon=1e-7):
        super(MLPGaussianPolicy, self).__init__(state_dim, hidden_sizes, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.bijector = TanhBijector(epsilon)
        self.apply(weights_init_)

    def forward(self, state):
        x = self.mlp(state)
        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))
        return mu, log_std.exp()

    def dist(self, state):
        mu, std = self.forward(state)
        return Normal(mu, std)

    def log_prob(self, state, action):
        dist = self.dist(state)
        x = self.bijector.inverse(action)
        return dist.log_prob(x).sum(dim=-1, keepdims=True)

    def entropy(self, state):
        dist = self.dist(state)
        return dist.entropy().mean()
    
    def sample(self, state):
        dist = self.dist(state)
        sample = dist.rsample() 
        log_prob = dist.log_prob(sample)
        log_prob -= self.bijector.log_prob_correction(sample)
        return self.bijector.forward(sample), log_prob.sum(1, keepdim=True)


# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py#L542
class MLPGaussianSDEPolicy(MLPBase):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU, log_std_init=-2.0, epsilon=1e-7):
        super(MLPGaussianSDEPolicy, self).__init__(state_dim, hidden_sizes, activation)
        self.epsilon = epsilon
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        log_std = torch.ones(hidden_sizes[-1], action_dim)
        self.log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        self.bijector = TanhBijector(epsilon)
        self._sample_weights(log_std)
        self.apply(weights_init_)

    def forward(self, state):
        x = self.mlp(state)
        self._latent_sde = x
        return self.mu_layer(x), self.log_std

    def dist(self, state):
        mu, log_std = self.forward(state)
        if self._latent_sde.dim() == 1:
            self._latent_sde = self._latent_sde.unsqueeze(dim=0)
            var = torch.mm(self._latent_sde ** 2, self._get_std(log_std) ** 2).squeeze(dim=0)
        else:
            var = torch.mm(self._latent_sde ** 2, self._get_std(log_std) ** 2)    
        return Normal(mu, torch.sqrt(var + self.epsilon))

    def log_prob(self, state, action):
        dist = self.dist(state)
        x = self.bijector.inverse(action)
        return dist.log_prob(x).sum(dim=-1, keepdims=True)

    def entropy(self, state):
        dist = self.dist(state)
        return dist.entropy().mean()
    
    def sample(self, state):
        dist = self.dist(state)
        sample = dist.rsample() 
        noise = self.get_noise()
        log_prob = dist.log_prob(sample)
        log_prob -= self.bijector.log_prob_correction(sample)
        return self.bijector.forward(sample + noise), log_prob.sum(1, keepdim=True)

    def _sample_weights(self, log_std):
        std = self._get_std(log_std)
        self.weights_dist = Normal(torch.zeros_like(std), std)
        self.exploration_mat = self.weights_dist.rsample()

    def _get_std(self, log_std):
        below_threshold = torch.exp(log_std) * (log_std <= 0)
        safe_log_std = log_std * (log_std > 0) + self.epsilon
        above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
        return below_threshold + above_threshold

    def get_noise(self):
        exploration_mat = self.exploration_mat.to(self._latent_sde.device)
        return torch.mm(self._latent_sde, exploration_mat).squeeze(dim=0)

    def reset_noise(self):
        self._sample_weights(self.log_std)

        
class MLPVFunction(MLPBase):
    def __init__(self, state_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        super(MLPVFunction, self).__init__(state_dim, hidden_sizes, activation)
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.apply(weights_init_)

    def forward(self, state):
        return self.output_layer(self.mlp(state))


class MLPQFunction(MLPBase):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        super(MLPQFunction, self).__init__(state_dim + action_dim, hidden_sizes, activation)
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.output_layer(self.mlp(x))


class MLPDoubleQFunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        super(MLPDoubleQFunction, self).__init__()
        self.q1 = MLPQFunction(state_dim, action_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(state_dim, action_dim, hidden_sizes, activation)
        self.apply(weights_init_)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)
    

class MLPQuantileQFunction(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles=25, n_nets=2, hidden_sizes=(64, 64), activation=nn.ReLU):
        super(MLPQuantileQFunction, self).__init__()
        self.nets = nn.ModuleList()
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        input_dim = state_dim + action_dim

        for _ in range(n_nets):
            net = MLPBase(input_dim, hidden_sizes, activation)
            self.nets.append(nn.Sequential(net, nn.Linear(hidden_sizes[-1], n_quantiles)))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        quantiles = torch.stack(tuple(net(x) for net in self.nets), dim=1)
        return quantiles
    







class MLPQNetwork(MLPBase):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU, dueling_mode=False):
        super(MLPQNetwork, self).__init__(state_dim, hidden_sizes, activation)
        self.dueling_mode = dueling_mode
        if dueling_mode:
            self.value_layer = nn.Linear(hidden_sizes[-1], 1)
            self.advantage_layer = nn.Linear(hidden_sizes[-1], action_dim)
        else:
            self.output_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = self.mlp(state)
        if self.dueling_mode:
            value = self.value_layer(x)
            advantage = self.advantage_layer(x)
            output = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        else:
            output = self.output_layer(x)
        return output

class MLPDoubleQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation=nn.ReLU, dueling_mode=False):
        super(MLPDoubleQNetwork, self).__init__()
        self.q1 = MLPQNetwork(state_dim, action_dim, hidden_sizes, activation, dueling_mode)
        self.q2 = MLPQNetwork(state_dim, action_dim, hidden_sizes, activation, dueling_mode)            
        self.apply(weights_init_)

    def forward(self, state):
        return self.q1(state), self.q2(state)

class MLPQuantileQNetwork(MLPBase):
    def __init__(self, state_dim, action_dim, n_quantiles=200, hidden_sizes=(64, 64), activation=nn.ReLU, dueling_mode=False):
        super(MLPQuantileQNetwork, self).__init__(state_dim, hidden_sizes, activation)
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.dueling_mode = dueling_mode
        if dueling_mode:
            self.value_quantile = nn.Linear(hidden_sizes[-1], 1)
            self.advantage_quantile = nn.Linear(hidden_sizes[-1], action_dim * n_quantiles)
        else:
            self.output_quantile = nn.Linear(hidden_sizes[-1], action_dim * n_quantiles)
        self.apply(weights_init_)

    def forward(self, state):
        x = self.mlp(state)
        if self.dueling_mode:
            value = self.value_quantile(x)
            advantage = self.advantage_quantile(x)
            quantiles = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        else:
            quantiles = self.output_quantile(x)
        return quantiles.view(-1, self.n_quantiles, self.action_dim)
    
class MLPRainbowQNetwork(MLPNoisyBase):
    def __init__(self, state_dim, action_dim, atom_size, support, hidden_sizes=(64, 64), activation=nn.ReLU):
        super(MLPRainbowQNetwork, self).__init__(state_dim, hidden_sizes, activation)
        self.atom_size = atom_size
        self.support = support
        self.action_dim = action_dim
        self.value_layer = NoisyLinear(hidden_sizes[-1], atom_size)
        self.advantage_layer = NoisyLinear(hidden_sizes[-1], action_dim * atom_size)
        self.apply(weights_init_)

    def forward(self, state):
        dist = self.dist(state)
        q = torch.sum(dist * self.support, dim=2)
        return q
    
    def dist(self, state):
        x = self.mlp(state)
        advantage = self.advantage_layer(x).view(-1, self.action_dim, self.atom_size)
        value = self.value_layer(x).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)
        return dist
    
    def reset_noise(self):
        self.value_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.base_reset_noise()
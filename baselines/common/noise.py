import torch
import numpy as np


class GaussianNoise:
    def __init__(self, size, mu=0.0, sigma=0.2, device='cpu'):
        self.size = (size, )
        self.mu = mu
        self.sigma = sigma
        self.device = device

    def sample(self):
        noise = torch.normal(mean=self.mu, std=self.sigma, size=self.size, device=self.device)
        return noise
    
    
class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, device='cpu'):
        self.size = (size, )
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.device = device
        self.reset()

    def reset(self):
        self.noise = torch.full(self.size, self.mu, device=self.device)
        
    def sample(self):
        x = self.noise
        dx = self.theta * (self.mu - x) * self.dt + \
             self.sigma * np.sqrt(self.dt) * torch.randn_like(x).to(self.device)
        self.noise = x + dx
        return self.noise
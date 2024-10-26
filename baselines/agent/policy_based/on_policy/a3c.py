import os, numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F, torch.multiprocessing as mp
from copy import deepcopy
from baselines.common.network import MLPGaussianPolicy, MLPVFunction
from baselines.common.buffer import RolloutBuffer
from baselines.common.optim import SharedAdam
from baselines.common.train import seed_all
from baselines.common.operation import GAE

class A3C:
    def __init__(
            self, 
            env,
            seed=0,
            actor_sizes=(64, 64),
            critic_sizes=(64, 64),
            pg_activation=torch.tanh,
            vf_activation=torch.tanh,
            actor_lr=3e-4,
            critic_lr=3e-4,
            n_steps=128,
            gamma=0.99,
            lmda=0.95,
            vf_coef=1.0,
            ent_coef=0.2,
            max_iters=int(1e+6),
            eval_freq=int(1e+4),
            eval_iters=10,
            n_envs=mp.cpu_count()
            ):
        
        os.environ["OMP_NUM_THREADS"] = "1"

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        global_actor = MLPGaussianPolicy(state_dim, action_dim, actor_sizes, pg_activation).to(device)
        global_actor.share_memory()
        global_critic = MLPVFunction(state_dim, critic_sizes, vf_activation).to(device)
        global_critic.share_memory()

        global_pg_optim = SharedAdam(global_actor.parameters(), lr=actor_lr)
        global_pg_optim.share_memory()
        global_vf_optim = SharedAdam(global_critic.parameters(), lr=critic_lr)
        global_vf_optim.share_memory()

        global_ep = mp.Value('i', 0)
        global_ep_ret = mp.Value('d', 0.0)
        self.ret_queue = mp.Queue()

        self.workers = [
            Worker(
                i, env, seed, state_dim, action_dim, device,
                actor_sizes, critic_sizes, pg_activation, vf_activation,
                global_actor, global_critic, global_pg_optim, global_vf_optim,
                global_ep, global_ep_ret, self.ret_queue, 
                n_steps, gamma, lmda, vf_coef, ent_coef,
                max_iters, eval_freq, eval_iters
            ) for i in range(n_envs)
        ]
    
    def train(self):
        for worker in self.workers:
            worker.start()

        rets = []
        timesteps = []
        while True:
            result = self.ret_queue.get()
            if result is not None:
                timestep, ret = result
                timesteps.append(timestep)
                rets.append(ret)
            else:
                break
        
        for worker in self.workers:
            worker.join()
        
        self.plot(timesteps, rets)

    def plot(self, timesteps, rets):
        plt.plot(timesteps, rets)
        plt.ylabel('Average Return')
        plt.xlabel('Timesteps')
        plt.title('A3C Training Progress')
        plt.show()


class Worker(mp.Process):
    def __init__(
            self, 
            name, env, seed, state_dim, action_dim, device,
            actor_sizes, critic_sizes, pg_activation, vf_activation,
            global_actor, global_critic, global_pg_optim, global_vf_optim,
            global_ep, global_ep_ret, ret_queue, 
            update_after, gamma, lmda, vf_coef, ent_coef,
            max_iters, eval_freq, eval_iters
            ):
        super(Worker, self).__init__()

        self.name = f'WORKER {name}'
        self.device = device

        self.env = deepcopy(env)
        self.seed = seed + 100 * name
        seed_all(seed + name)

        self.global_actor = global_actor
        self.global_critic = global_critic
        self.global_pg_optim = global_pg_optim
        self.global_vf_optim = global_vf_optim
        self.global_ep = global_ep
        self.global_ep_ret = global_ep_ret
        self.ret_queue = ret_queue

        self.local_actor = MLPGaussianPolicy(state_dim, action_dim, actor_sizes, pg_activation).to(self.device)
        self.local_critic = MLPVFunction(state_dim, critic_sizes, vf_activation).to(self.device)
        self.buffer = RolloutBuffer(self.device)

        self.update_after = update_after
        self.gamma = gamma
        self.lmda = lmda
        self.vf_coef = vf_coef 
        self.ent_coef = ent_coef

        self.max_iters = max_iters
        self.eval_freq = eval_freq
        self.eval_iters = eval_iters
        self.max_action = self.env.action_space.high
        self.min_action = self.env.action_space.low

    @torch.no_grad()
    def act(self, state, training=True):
        self.local_actor.train(training)
        state = torch.FloatTensor(state).to(self.device)
        mu, std = self.local_actor(state)
        action = torch.normal(mu, std) if training else mu
        return torch.tanh(action).cpu().numpy()
    
    def learn(self, states, actions, rewards, next_states, dones):
        self.global_actor.train()
        self.global_critic.train()

        with torch.no_grad():
            values = self.local_critic(states)
            next_values = self.local_critic(next_states)
            rets, advs = GAE(values, next_values, rewards, dones, self.gamma, self.lmda)

        log_probs = self.local_actor.log_prob(states, actions)
        pg_loss = -(log_probs * advs).mean()
        vf_loss = F.mse_loss(self.local_critic(states), rets)
        entropy = self.local_actor.entropy(states)
        loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

        self.global_pg_optim.zero_grad()
        self.global_vf_optim.zero_grad()
        loss.backward()

        for local_params, global_params in zip(self.local_actor.parameters(), self.global_actor.parameters()):
            global_params._grad = local_params.grad
        
        for local_params, global_params in zip(self.local_critic.parameters(), self.global_critic.parameters()):
            global_params._grad = local_params.grad

        self.global_pg_optim.step()
        self.global_vf_optim.step()

        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())

    def evaluate(self):
        total_score = 0.0
        for _ in range(self.eval_iters):
            rand_seed = self.seed + np.random.randint(0, 1000)
            state = self.env.reset(seed=rand_seed)
            score, done = 0.0, False
            
            while not done:
                action = self.act(state, training=False)
                x = 0.5 * (action + 1) * (self.max_action - self.min_action) + self.min_action
                next_state, reward, done, _ = self.env.step(x)

                score += reward
                state = next_state

            total_score += score
        return total_score / self.eval_iters
    
    def run(self):
        state = self.env.reset()
        for timestep in range(1, self.max_iters + 1):
            action = self.act(state)
            x = 0.5 * (action + 1) * (self.max_action - self.min_action) + self.min_action
            next_state, reward, done, _ = self.env.step(x)

            self.buffer.store(state, action, reward, next_state, done)
            state = next_state

            if done:
                state = self.env.reset()

            if self.buffer.size >= self.update_after:
                states, actions, rewards, next_states, dones = self.buffer.sample()
                self.learn(states, actions, rewards, next_states, dones)

            if timestep % self.eval_freq == 0:
                avg_return = self.evaluate()
                self.record(avg_return)

        self.ret_queue.put(None)

    def record(self, avg_return):
        with self.global_ep.get_lock():
            self.global_ep.value += self.eval_freq

        with self.global_ep_ret.get_lock():
            self.global_ep_ret.value = avg_return

        self.ret_queue.put((self.global_ep.value, self.global_ep_ret.value))
        print(f'{self.name} | TIMESTEPS: {self.global_ep.value},   RETURN: {round(self.global_ep_ret.value, 2)}')

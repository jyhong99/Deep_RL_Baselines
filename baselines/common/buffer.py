import ray, torch, random, threading, numpy as np
from baselines.common.operation import MinSegmentTree, SumSegmentTree


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class RolloutBuffer:
    def __init__(self, device):
        self.buffer = list()
        self.device = device

    def store(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, reward_norm=True, epsilon=1e-8):
        state, action, reward, next_state, done = map(np.array, zip(*self.buffer))
        self.buffer.clear()
        
        states = torch.FloatTensor(state).to(self.device)
        actions = torch.FloatTensor(action).to(self.device)
        rewards = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_state).to(self.device)
        dones = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        if reward_norm:
            rewards = (rewards - rewards.mean()) / (rewards.std() + epsilon)
        return states, actions, rewards, next_states, dones
    
    @property
    def size(self):
        return len(self.buffer)


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, device):
        self.states = np.zeros(combined_shape(max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros(combined_shape(max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

        self.device = device
        self.ptr, self.size, self.max_size = 0, 0, max_size
    
    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, reward_norm=True, epsilon=1e-8):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.FloatTensor(self.states[idxs]).to(self.device)
        actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idxs]).to(self.device)
        dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
    
        if reward_norm:
            rewards = (rewards - rewards.mean()) / (rewards.std() + epsilon)
        return states, actions, rewards, next_states, dones


class PrioritizedReplayBuffer(ReplayBuffer):    
    def __init__(self, state_dim, action_dim, max_size, device, alpha=0.6):
        super(PrioritizedReplayBuffer, self).__init__(state_dim, action_dim, max_size, device)
        self.max_prio, self.tree_ptr = 1.0, 0
        self.alpha = alpha
    
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(self, state, action, reward, next_state, done):
        super().store(state, action, reward, next_state, done)
        self.sum_tree[self.tree_ptr] = self.max_prio ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_prio ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample(self, batch_size, beta=0.4, reward_norm=True, epsilon=1e-8):
        self.batch_size = batch_size
        idxs = self._sample_proportional()
        
        states = torch.FloatTensor(self.states[idxs]).to(self.device)
        actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idxs]).to(self.device)
        dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
        weights = torch.FloatTensor([self._calculate_weight(i, beta) for i in idxs]).unsqueeze(1).to(self.device)

        if reward_norm:
            rewards = (rewards - rewards.mean()) / (rewards.std() + epsilon)
        return states, actions, rewards, next_states, dones, weights, idxs
        
    def update_priorities(self, idxs, prios):
        for idx, prio in zip(idxs, prios):
            self.sum_tree[idx] = prio ** self.alpha
            self.min_tree[idx] = prio ** self.alpha
            self.max_prio = max(self.max_prio, prio)

    def _sample_proportional(self):
        idxs = []
        p_total = self.sum_tree.sum(0, self.size - 1)
        segment = p_total / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            idxs.append(idx)
            
        return idxs
    
    def _calculate_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size) ** (-beta)
        
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.size) ** (-beta)
        weight = weight / max_weight
        
        return weight








@ray.remote(num_gpus=0.1)
class SharedRolloutBuffer:
    def __init__(self, device):
        self.buffer = list()
        self.device = device

    @property
    def size(self):
        return len(self.buffer)
    
    def get_size(self):
        return self.size
    
    def store(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, reward_norm=True, epsilon=1e-8):
        state, action, reward, next_state, done = map(np.array, zip(*self.buffer))
        self.buffer.clear()
        
        states = torch.FloatTensor(state).to(self.device)
        actions = torch.FloatTensor(action).to(self.device)
        rewards = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_state).to(self.device)
        dones = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        if reward_norm:
            rewards = (rewards - rewards.mean()) / (rewards.std() + epsilon)
        return states, actions, rewards, next_states, dones
    

@ray.remote(num_gpus=0.1)
class SharedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, device):
        self.states = np.zeros(combined_shape(max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros(combined_shape(max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

        self.lock = threading.Lock()
        self.device = device
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def get_size(self):
        return self.size
        
    def store(self, state, action, reward, next_state, done):
        with self.lock: 
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr] = done

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, reward_norm=True, epsilon=1e-8):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.FloatTensor(self.states[idxs]).to(self.device)
        actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idxs]).to(self.device)
        dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
    
        if reward_norm:
            rewards = (rewards - rewards.mean()) / (rewards.std() + epsilon)
        return states, actions, rewards, next_states, dones
    

@ray.remote(num_gpus=0.1)
class SharedPrioritizedReplayBuffer(ReplayBuffer):    
    def __init__(self, state_dim, action_dim, max_size, device, alpha=0.6):
        self.states = np.zeros(combined_shape(max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros(combined_shape(max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

        self.device = device
        self.ptr, self.size, self.max_size = 0, 0, max_size
        self.max_prio, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.lock = threading.Lock()

        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def get_size(self):
        return self.size
            
    def store(self, state, action, reward, next_state, done):
        with self.lock: 
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr] = done

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

            self.sum_tree[self.tree_ptr] = self.max_prio ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_prio ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample(self, batch_size, beta=0.4, reward_norm=True, epsilon=1e-8):
        self.batch_size = batch_size
        idxs = self._sample_proportional()
        
        states = torch.FloatTensor(self.states[idxs]).to(self.device)
        actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idxs]).to(self.device)
        dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
        weights = torch.FloatTensor([self._calculate_weight(i, beta) for i in idxs]).unsqueeze(1).to(self.device)

        if reward_norm:
            rewards = (rewards - rewards.mean()) / (rewards.std() + epsilon)
        return states, actions, rewards, next_states, dones, weights, idxs
        
    def update_priorities(self, idxs, prios):
        for idx, prio in zip(idxs, prios):
            self.sum_tree[idx] = prio ** self.alpha
            self.min_tree[idx] = prio ** self.alpha
            self.max_prio = max(self.max_prio, prio)

    def _sample_proportional(self):
        idxs = []
        p_total = self.sum_tree.sum(0, self.size - 1)
        segment = p_total / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            idxs.append(idx)

        return idxs
    
    def _calculate_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.size) ** (-beta)
        weight = weight / max_weight
    
        return weight
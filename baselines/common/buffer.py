import ray, torch, random, pickle, numpy as np
from baselines.common.operation import MinSegmentTree, SumSegmentTree


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class BaseBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, device, reward_norm, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.reward_norm = reward_norm
        self.epsilon = epsilon
        self.reset()

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.ptr = 0
            self.is_full = True
        
    def sample(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        self.states = np.zeros(combined_shape(self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros(combined_shape(self.buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.ptr, self.is_full = 0, False

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, load_path):
        with open(load_path, 'rb') as f:
            buffer = pickle.load(f)
        return buffer
    
    @property
    def size(self):
        return self.buffer_size if self.is_full else self.ptr
    
    def _normalize_elements(self, elements):
        mean = elements.mean()
        std = elements.std() + self.epsilon
        normalized_elements = (elements - mean) / std
        return normalized_elements








class RolloutBuffer(BaseBuffer):
    def __init__(self, state_dim, action_dim, buffer_size, device, reward_norm=False, epsilon=1e-8):
        super().__init__(
            state_dim=state_dim, 
            action_dim=action_dim, 
            buffer_size=buffer_size, 
            batch_size=None, 
            device=device, 
            reward_norm=reward_norm, 
            epsilon=epsilon
            )

    def sample(self):
        states = torch.FloatTensor(self.states[:self.ptr]).to(self.device)
        actions = torch.FloatTensor(self.actions[:self.ptr]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[:self.ptr]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[:self.ptr]).to(self.device)
        dones = torch.FloatTensor(self.dones[:self.ptr]).to(self.device)

        self.reset()
        if self.reward_norm:
            rewards = self._normalize_elements(rewards)

        return states, actions, rewards, next_states, dones


class ReplayBuffer(BaseBuffer):
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, device, reward_norm=False, epsilon=1e-8):
        super().__init__(
            state_dim=state_dim, 
            action_dim=action_dim, 
            buffer_size=buffer_size, 
            batch_size=batch_size, 
            device=device, 
            reward_norm=reward_norm, 
            epsilon=epsilon
            )

    def sample(self):
        idxs = np.random.randint(0, self.ptr, size=self.batch_size)
        states = torch.FloatTensor(self.states[idxs]).to(self.device)
        actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idxs]).to(self.device)
        dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
    
        if self.reward_norm:
            rewards = self._normalize_elements(rewards)

        return states, actions, rewards, next_states, dones


class PrioritizedReplayBuffer(ReplayBuffer):    
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, device, alpha=0.6, reward_norm=False, epsilon=1e-8):
        super().__init__(
            state_dim=state_dim, 
            action_dim=action_dim, 
            buffer_size=buffer_size, 
            batch_size=batch_size, 
            device=device, 
            reward_norm=reward_norm, 
            epsilon=epsilon
            )
        
        self.max_prio, self.tree_ptr = 1., 0
        self.alpha = alpha
    
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(self, state, action, reward, next_state, done):
        super().store(state, action, reward, next_state, done)
        self.sum_tree[self.tree_ptr] = self.max_prio ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_prio ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size

    def sample(self, beta):
        idxs = self._sample_proportional()
        states = torch.FloatTensor(self.states[idxs]).to(self.device)
        actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idxs]).to(self.device)
        dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
        weights = torch.FloatTensor([self._calculate_weight(i, beta) for i in idxs]).unsqueeze(1).to(self.device)

        if self.reward_norm:
            rewards = self._normalize_elements(rewards)

        return states, actions, rewards, next_states, dones, weights, idxs
        
    def update_priorities(self, idxs, prios):
        for idx, prio in zip(idxs, prios):
            self.sum_tree[idx] = prio ** self.alpha
            self.min_tree[idx] = prio ** self.alpha
            self.max_prio = max(self.max_prio, prio)

    def _sample_proportional(self):
        idxs = []
        p_total = self.sum_tree.sum(0, self.ptr - 1)
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
        max_weight = (p_min * self.ptr) ** (-beta)
        
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.ptr) ** (-beta)
        weight = weight / max_weight
        
        return weight








@ray.remote(num_gpus=0.1)
class SharedRolloutBuffer(object):
    def __init__(self, state_dim, action_dim, buffer_size, device, reward_norm, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.buffer_size = buffer_size
        self.reward_norm = reward_norm
        self.epsilon = epsilon
        self.reset()

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.ptr = 0
            self.is_full = True
        
    def sample(self):
        states = torch.FloatTensor(self.states[:self.ptr]).to(self.device)
        actions = torch.FloatTensor(self.actions[:self.ptr]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[:self.ptr]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[:self.ptr]).to(self.device)
        dones = torch.FloatTensor(self.dones[:self.ptr]).to(self.device)

        self.reset()
        if self.reward_norm:
            rewards = self._normalize_elements(rewards)

        return states, actions, rewards, next_states, dones

    def reset(self):
        self.states = np.zeros(combined_shape(self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros(combined_shape(self.buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.ptr, self.is_full = 0, False

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, load_path):
        with open(load_path, 'rb') as f:
            buffer = pickle.load(f)
        return buffer
    
    def size(self):
        return self.buffer_size if self.is_full else self.ptr
    
    def _normalize_elements(self, elements):
        mean = elements.mean()
        std = elements.std() + self.epsilon
        normalized_elements = (elements - mean) / std
        return normalized_elements
    

    

@ray.remote(num_gpus=0.1)
class SharedReplayBuffer(object):
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, device, alpha=0.6, reward_norm=False, epsilon=1e-8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.reward_norm = reward_norm
        self.epsilon = epsilon
        self.reset()

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.ptr = 0
            self.is_full = True
        
    def sample(self):
        idxs = np.random.randint(0, self.ptr, size=self.batch_size)
        states = torch.FloatTensor(self.states[idxs]).to(self.device)
        actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idxs]).to(self.device)
        dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
    
        if self.reward_norm:
            rewards = self._normalize_elements(rewards)

        return states, actions, rewards, next_states, dones

    def reset(self):
        self.states = np.zeros(combined_shape(self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros(combined_shape(self.buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.ptr, self.is_full = 0, False

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, load_path):
        with open(load_path, 'rb') as f:
            buffer = pickle.load(f)
        return buffer
    
    def size(self):
        return self.buffer_size if self.is_full else self.ptr
    
    def _normalize_elements(self, elements):
        mean = elements.mean()
        std = elements.std() + self.epsilon
        normalized_elements = (elements - mean) / std
        return normalized_elements
    



@ray.remote(num_gpus=0.1)
class SharedPrioritizedReplayBuffer(object):  
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, device, alpha, reward_norm, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.reward_norm = reward_norm
        self.epsilon = epsilon

        self.max_prio, self.tree_ptr = 1., 0
        self.alpha = alpha
    
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

        self.reset()

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.ptr = 0
            self.is_full = True

        self.sum_tree[self.tree_ptr] = self.max_prio ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_prio ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size

    def sample(self, beta):
        idxs = self._sample_proportional()
        states = torch.FloatTensor(self.states[idxs]).to(self.device)
        actions = torch.FloatTensor(self.actions[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idxs]).to(self.device)
        dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
        weights = torch.FloatTensor([self._calculate_weight(i, beta) for i in idxs]).unsqueeze(1).to(self.device)

        if self.reward_norm:
            rewards = self._normalize_elements(rewards)

        return states, actions, rewards, next_states, dones, weights, idxs

    def reset(self):
        self.states = np.zeros(combined_shape(self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros(combined_shape(self.buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.ptr, self.is_full = 0, False

    def update_priorities(self, idxs, prios):
        for idx, prio in zip(idxs, prios):
            self.sum_tree[idx] = prio ** self.alpha
            self.min_tree[idx] = prio ** self.alpha
            self.max_prio = max(self.max_prio, prio)

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, load_path):
        with open(load_path, 'rb') as f:
            buffer = pickle.load(f)
        return buffer

    def size(self):
        return self.buffer_size if self.is_full else self.ptr
    
    def _normalize_elements(self, elements):
        mean = elements.mean()
        std = elements.std() + self.epsilon
        normalized_elements = (elements - mean) / std
        return normalized_elements
    
    def _sample_proportional(self):
        idxs = []
        p_total = self.sum_tree.sum(0, self.ptr - 1)
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
        max_weight = (p_min * self.ptr) ** (-beta)
        
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.ptr) ** (-beta)
        weight = weight / max_weight
        
        return weight
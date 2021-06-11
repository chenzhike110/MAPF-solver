import numpy as np
import random

from numpy.core.defchararray import index
from numpy.core.fromnumeric import size

class linear_schedule:
    def __init__(self, total_timesteps, final_ratio, init_ratio=1.0):
        self.total_timesteps = total_timesteps
        self.final_ratio = final_ratio
        self.init_ratio = init_ratio
    
    def get_value(self, timestep):
        frac = min(float(timestep) / self.total_timesteps, 1.0)
        return self.init_ratio - frac * (self.init_ratio - self.final_ratio)
    
def select_action(action_value, explore_eps):
    action_value = action_value.cpu().numpy().squeeze()
    if random.random() > explore_eps:
        try:
            action = np.argmax(action_value, axis=1)  
        except:
            action = np.argmax(action_value, axis=0)  
    else:
        try:
            action = np.random.randint(0,5,(action_value.shape[0]))
        except:
            action = int(np.random.randint(0,5,size=(1)))
    return action

class reward_recoder:
    def __init__(self, num, history_length=1000):
        self.history_length = history_length
        self.buffer = [[0.0] for _ in range(num)]
        self._episode_length = 1
    
    def add_reward(self, reward, index):
        self.buffer[index][-1] += reward
    
    def start_new_episode(self):
        for i in range(len(self.buffer)):
            if self.get_length > self.history_length:
                self.buffer[i].pop(0)
            self.buffer[i].append(0.0)
        self._episode_length += 1
    
    @property
    def get_length(self):
        return max(len(i) for i in self.buffer)
    
    @property
    def mean(self):
        return np.mean(np.array(self.buffer[0]))
    
    @property
    def latest(self):
        return [i[-1] for i in self.buffer]
    
    @property
    def success_rate(self):
        return sum(np.array(self.buffer[0][-100:])>15)
    
    @property
    def num_episodes(self):
        return self._episode_length

class replay_buffer:
    def __init__(self, memory_size):
        self.storge = []
        self.memory_size = memory_size
        self.next_idx = 0
    
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storge):
            self.storge.append(data)
        else:
            self.storge[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size
    
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storge[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)
    
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storge)-1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
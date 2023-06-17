import numpy as np
from src.configuration.config import *

class ReplayBuffer(object):
	"""docstring for ReplayBuffer"""
	mem_counter = 0 
	def __init__(self, capacity):
		super(ReplayBuffer, self).__init__()
		self.capacity = capacity

		self.states = np.zeros((capacity, N_OBSERVATION), dtype=np.float32)
		self.actions = np.zeros(capacity, dtype=np.int64)
		self.rewards = np.zeros(capacity, dtype=np.float32)
		self.next_states = np.zeros((capacity, N_OBSERVATION),dtype=np.float32)
		self.dones = np.zeros(capacity, dtype=np.bool)


	def add_experience(self, state, action, reward, next_state, done):
		cur_mem_index = self.mem_counter % self.capacity
		self.states[cur_mem_index] = state
		self.actions[cur_mem_index] = action
		self.rewards[cur_mem_index] = reward
		self.next_states[cur_mem_index] = next_state
		self.dones[cur_mem_index] = done

		self.mem_counter += 1

	def sample_experience(self):
		max_size = min(self.mem_counter, self.capacity)
		batch_indices = np.random.choice(max_size, BATCH_SIZE, replace=True)

		states = self.states[batch_indices]
		actions = self.actions[batch_indices]
		rewards = self.rewards[batch_indices]
		next_states = self.next_states[batch_indices]
		dones = self.dones[batch_indices]

		return (states, actions, rewards, next_states, dones)


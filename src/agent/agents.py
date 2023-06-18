import random
import numpy as np 
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from src.storage.replay_buffer import ReplayBuffer
from src.configuration.config import *
from src.network.network import DqnNetwork

# set the seed value


import platform
from itertools import count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DQNAgent(object):
	"""docstring for DQNAgent"""
	device = 'mps' if torch.backends.mps.is_available() else 'cpu'
	if platform.system() == "Windows":
		device = 'cuda'if torch.cuda.is_available() else 'cpu'

	def __init__(self):
		super(DQNAgent, self).__init__()
		self.memory = ReplayBuffer(MEMORY_SIZE)
		self.policy_net = DqnNetwork().to(self.device)
		self.target_net = DqnNetwork().to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.epsilon_threshold = EPSILON

		self.writter = SummaryWriter(comment="")

		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
		random.seed(SEED)
		np.random.seed(SEED)
		torch.manual_seed(SEED)

	def reduce_exploration(self):
		self.epsilon_threshold = max(EPSILON_END, EPSILON_DECAY_RATE*self.epsilon_threshold)

	def get_action(self, observation, isTrining=True):
		if isTrining:
			exploit_threshold = random.random()
			if exploit_threshold <= self.epsilon_threshold:
				return np.random.choice(N_ACTION)
			else:
				state = torch.FloatTensor(observation).to(self.device)
				action  = self.policy_net(state).argmax().item()
				return action
		else:
			state = torch.FloatTensor(observation).to(self.device)
			return self.policy_net(state).argmax().item()


	def learn_policy(self):
		if self.memory.mem_counter < BATCH_SIZE:
			return
		states, actions, rewards, next_states, dones = self.memory.sample_experience()
		states = torch.FloatTensor(states).to(self.device)
		actions = torch.LongTensor(actions.reshape(-1,1)).to(self.device)
		rewards = torch.FloatTensor(rewards.reshape(-1,1)).to(self.device)
		next_states = torch.FloatTensor(next_states).to(self.device)
		dones = torch.FloatTensor(dones.reshape(-1,1)).to(self.device)

		q_values = self.policy_net(states).gather(1,actions)
		q_target_max = self.target_net(next_states).max(1)[0].unsqueeze(1).detach()
		target_q_values = rewards + GAMMA*q_target_max*dones
		criterion = torch.nn.SmoothL1Loss()
		loss = criterion(q_values, target_q_values)
		# update network
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


	def update_target_network(self):
		#self.target_net.load_state_dict(self.policy_net.state_dict())
		# soft update of the target network's weights
		# θ′ ← τ θ + (1 −τ )θ′
		target_net_state_dict = self.target_net.state_dict()
		policy_net_state_dict = self.policy_net.state_dict()
		for key in policy_net_state_dict:
			target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
		self.target_net.load_state_dict(target_net_state_dict)


	def train_RL(self, env):
		max_reward = 0.0
		for e in tqdm(range(1000),colour="red"):
			state, info = env.reset()
			r_r = 0
			for t in count():
				action = self.get_action(state)
				next_state, reward, done, time_loss, _ = env.step(action)
				done_mask = 0.0 if done else 1.0
				self.memory.add_experience(state,action,reward/10.0,next_state,done)
				if done:
					break
				state = next_state
				r_r += reward
				self.learn_policy()
				env.move_gui()
			if (e+1)%TARGET_NET_UPDATE_FRE == 0:
				self.update_target_network()
			self.reduce_exploration()
			print(f'Episode: {e+1}/{self.epsilon_threshold}')
			self.writter.add_scalar("Reward/Train", r_r, (e+1))
			env.closeEnvConnection()






import random
import numpy as np 
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from src.storage.prioritized_experience_replay_memory import PER
from src.configuration.config import *
from src.network.network import DqnNetwork

# set the seed value


import platform
from itertools import count
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

class DQNPERAgent(object):
	"""docstring for DQNPERAgent"""
	device = 'mps' if torch.backends.mps.is_available() else 'cpu'
	if platform.system() == "Windows":
		device = 'cuda'if torch.cuda.is_available() else 'cpu'

	def __init__(self, process=""):
		super(DQNPERAgent, self).__init__()
		self.memory = PER(MEMORY_SIZE)
		self.policy_net = DqnNetwork().to(self.device)
		self.target_net = DqnNetwork().to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.epsilon_threshold = EPSILON
		self.writter = SummaryWriter(comment="PER_REPLAY"+process)
		self.ep_loss = 0.0

		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
		random.seed(SEED)
		np.random.seed(SEED)
		torch.manual_seed(SEED)

	def add_sample(self, state, action, reward, nxt_state,done):
		state_ = torch.FloatTensor(state).to(self.device)
		action_ = torch.LongTensor([action]).to(self.device)
		reward_ = torch.FloatTensor([reward]).to(self.device)
		nxt_state_ = torch.FloatTensor(nxt_state).to(self.device)
		done_ = torch.FloatTensor([done]).to(self.device)
		q_values = self.policy_net(state_).gather(0,action_)
		q_target_max = self.target_net(nxt_state_).max(0)[0].unsqueeze(0).detach()
		target_q_values = reward_ + GAMMA*q_target_max*done_
		error = abs(q_values-target_q_values).detach().item()
		self.memory.add(error, (state, action, reward, nxt_state,done))

		

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
		if self.memory.get_mem_size() < BATCH_SIZE:
			return
		batch_data,idxs, is_weights = self.memory.sample_experience(BATCH_SIZE)
		states = np.array([batch_data[i][0] for i in range(len(batch_data))])
		actions = np.array([batch_data[i][1] for i in range(len(batch_data))])
		rewards = np.array([batch_data[i][2] for i in range(len(batch_data))])
		next_states = np.array([batch_data[i][3] for i in range(len(batch_data))])
		dones = np.array([batch_data[i][4] for i in range(len(batch_data))])
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
		self.ep_loss += loss
		# update network
		self.ep_loss += loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		#time.sleep(5.0)


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
				next_state, reward, done, _ = env.step(action)
				done_mask = 0.0 if done else 1.0
				self.add_sample(state,action,reward,next_state,done_mask)
				if done:
					print(f'Done: {done}')
					break
				state = next_state
				r_r += reward
				self.learn_policy()
				env.move_gui()
			if (e+1)%TARGET_NET_UPDATE_FRE == 0:
				self.update_target_network()
			self.reduce_exploration()
			print(f'Episode: {e+1}/{self.epsilon_threshold} -> Reward: {r_r}')
			self.writter.add_scalar("Reward/Train", r_r, (e+1))
			self.writter.add_scalar("Loss/Train", self.ep_loss, (e+1))
			self.ep_loss = 0.0
			env.closeEnvConnection()






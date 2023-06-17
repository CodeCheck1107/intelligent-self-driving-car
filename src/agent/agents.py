import random
import numpy as np 
import torch
from torch.utils.tensorboard import SummaryWriter
from src.storage.replay_buffer import ReplayBuffer
from src.configuration.config import *
from src.network.network import DqnNetwork

# set the seed value
random.seed(SEED)
torch.manual_seed(SEED)

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
		self.policy_net.eval()
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()
		self.epsilon_threshold = EPSILON

		self.writter = SummaryWriter(comment="")


	def get_action(self, observation, isTrining=True):
		if isTrining:
			exploit_threshold = random.random()
			if exploit_threshold <= self.epsilon_threshold and self.epsilon_threshold > EPSILON_END:
				return np.random.choice(N_ACTION)
			else:
				state = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
				action  = self.policy_net(state).max(1)[1].item()
				return action
		else:
			state = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
			return self.policy_net(state).max(1)[1].item()


	def learn_policy(self):
		if self.memory.mem_counter < BATCH_SIZE:
			return
		states, actions, rewards, next_states, dones = self.memory.sample_experience()
		states = torch.FloatTensor(states).to(self.device)
		actions = torch.LongTensor(actions).to(self.device)
		rewards = torch.FloatTensor(rewards).to(self.device)
		next_states = torch.FloatTensor(next_states).to(self.device)
		dones = torch.BoolTensor(dones).to(self.device)
		state_action_values = self.policy_net(states).gather(1,actions.view(1, BATCH_SIZE))
		next_state_action_values = self.target_net(next_states).max(1)[0]
		q_target = rewards+GAMMA*next_state_action_values*~dones
		self.policy_net.optimizer.zero_grad()
		loss = self.policy_net.loss(q_target.view(1,BATCH_SIZE),state_action_values)
		loss.backward()
		self.policy_net.optimizer.step()
		self.epsilon_threshold *= EPSILON_DECAY_RATE







	def train_RL(self, env):
		max_reward = 0.0
		for e in tqdm(range(1000),colour="red"):
			state, info = env.reset()
			r_r = 0
			for t in count():
				action = self.get_action(state)
				next_state, reward, done, time_loss, _ = env.step(action)
				self.memory.add_experience(state,action,reward,next_state,done)
				if done:
					break
				else:
					pass
					#next_state = torch.tensor(observation,dtype=torch.float32, device=self.device).unsqueeze(0)

				state = next_state
				r_r += reward
				self.learn_policy()
				env.move_gui()
			self.writter.add_scalar("Reward/Train", r_r, (e+1))
			env.closeEnvConnection()






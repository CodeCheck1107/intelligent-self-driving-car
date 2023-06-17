import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.configuration.config import *
class DqnNetwork(nn.Module):
	"""docstring for DqnNetwork"""
	def __init__(self):
		super(DqnNetwork, self).__init__()
		self.n_observations = N_OBSERVATION
		self.n_actions = N_ACTION

		self.ln1 = nn.Linear(self.n_observations, 256)
		self.ln2 = nn.Linear(256,16)
		self.lout = nn.Linear(16, self.n_actions)
		self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
		self.loss = nn.MSELoss()

	def forward(self, x):
		x = F.relu(self.ln1(x))
		x = F.relu(self.ln2(x))
		return self.lout(x)





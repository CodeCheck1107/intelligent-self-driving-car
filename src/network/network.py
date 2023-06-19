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

		self.ln1 = nn.Linear(self.n_observations, 512)
		self.ln2 = nn.Linear(512,1024)
		self.ln3 = nn.Linear(1024,2048)
		self.ln4 = nn.Linear(2048,1024)
		self.lout = nn.Linear(1024, self.n_actions)

	def forward(self, x):
		x = F.relu(self.ln1(x))
		x = F.relu(self.ln2(x))
		x = F.relu(self.ln3(x))
		x = F.relu(self.ln4(x))
		return self.lout(x)





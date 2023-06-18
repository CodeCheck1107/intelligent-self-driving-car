import gym
import gym_sumo

from src.agent.agents import DQNAgent
def main():
	agent = DQNAgent()
	env = gym.make("sumo-v0", render_mode="")
	agent.train_RL(env) # for training



if __name__ == '__main__':
	main()
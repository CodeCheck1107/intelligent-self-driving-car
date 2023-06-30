import gym
import gym_sumo
from dqn_sumo_gym import Agent
import multiprocessing as mp
from multiprocessing import Pool, Process
def main(lr):
    agent = Agent("Agent",hp=lr)
    env = gym.make("sumo-v0", render_mode="")
    agent.train_RL(env) # for training
    #agent.test_RL(env)

if __name__ == '__main__':
    # lr = [0.01,0.001,0.0001,0.00001]
    # expoloration = [1000,10000,100000]
    # batch_size = [16,24,32,48,64]
    # rp_m = [5000,10000,20000,30000,50000]
    # proces = []
    # for i in rp_m:
    #     proc = Process(target=main, args=(i,))
    #     proces.append(proc)
    #     proc.start()

    # # complete
    # for proc in proces:
    #     proc.join()



    agent = Agent("Agent")
    env = gym.make("sumo-v0", render_mode="human")
    agent.train_RL(env) # for training
    #agent.test_RL(env)

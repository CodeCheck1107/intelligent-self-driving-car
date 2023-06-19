N_OBSERVATION = 21
N_ACTION = 5
MEMORY_SIZE = 500000



# seed for reproducibility
SEED = 0

# Hyperparameters
BATCH_SIZE = 2048
GAMMA = 0.99
EPSILON = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_RATE = 0.985
LEARNING_RATE = 1e-3
TAU=1e-2
TARGET_NET_UPDATE_FRE = 10 # episodes

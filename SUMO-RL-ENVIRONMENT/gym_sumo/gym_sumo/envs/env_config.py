NUM_OF_LANES = 5
RL_SENSING_RADIUS = 50.0
RL_MAX_SPEED_LIMIT = 50.0
RL_MIN_SPEED_LIMIT = 30.0
RL_ACC_RANGE = 2.6
RL_DCE_RANGE = 9.0
MIN_LANE_DENSITY = 0
MAX_LANE_DENSITY = 100 #this is an assumption for the traffic

EGO_ID = "av_0"

W1 = 0.1 # efficiency Reward
W2 = 0.001 # Collision reward
W3 = 0.001 # lane change Reward

# SIMULATION STEP
STEP_LENGTH=0.2
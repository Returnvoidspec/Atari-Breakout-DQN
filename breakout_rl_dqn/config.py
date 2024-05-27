# Hyperparameters as described in the paper

# Number of training cases over which each SGD update is computed
MINIBATCH_SIZE = 32

# SGD updates are sampled from this number of most recent frames
REPLAY_MEMORY_SIZE = 700_000

# The number of most recent frames experienced by the agent that are given as input to the Q network
AGENT_HISTORY_LENGTH = 4

# The frequency with which the target network is updated
TARGET_NETWORK_UPDATE_FREQUENCY = 10_000

# Discount factor gamma used in the Q-learning update
DISCOUNT_FACTOR = 0.99

# Repeat each action selected by the agent this many times
ACTION_REPEAT = 4

# The number of actions selected by the agent between successive SGD updates
UPDATE_FREQUENCY = 4

# The learning rate used by RMSProp
LEARNING_RATE = 0.00025

# Gradient momentum used by RMSProp
GRADIENT_MOMENTUM = 0.95

# Squared gradient (denominator) momentum used by RMSProp
SQUARED_GRADIENT_MOMENTUM = 0.95

# Constant added to the squared gradient in the denominator of the RMSProp update
MIN_SQUARED_GRADIENT = 0.01

# Initial value of ε in ε-greedy exploration
INITIAL_EXPLORATION = 1

# Final value of ε in ε-greedy exploration
FINAL_EXPLORATION = 0.1

# The number of frames over which the value of ε is linearly annealed to its final value
FINAL_EXPLORATION_FRAME = 1_000_000

# A uniform random policy is executed for this number of frames before learning starts
REPLAY_START_SIZE = 50_000

# Maximum number of 'do nothing' actions to be performed by the agent at the start of an episode
NO_OP_MAX = 30

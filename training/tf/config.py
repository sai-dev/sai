#!/usr/bin/env python3

# Board size and number of squares in a board. Board size must be an odd number!
BOARD_SIZE = 7
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE

# Channels in 1x1 convolution of heads. Default is 2 for policy and 1 for value
OUTPUTS_POLICY = 2
OUTPUTS_VALUE = 1

# Learning rate
LEARN_RATE = 0.05

# Sane values are from 4096 to 64 or so. The maximum depends on the amount
# of RAM in your GPU and the network size. You need to adjust the learning rate
# if you change this.
BATCH_SIZE = 512

# Use a random sample of 1/16th of the input data read. This helps
# improve the spread of games in the shuffle buffer.
DOWN_SAMPLE = 16

# Outputs new network after the specified number of training steps
TRAINING_STEPS = 2000 #2000 for AGZ, 500 for AZ

# Display intermediate output after the specified number of training steps
TRAINING_STEPS_VERBOSE = 500 #500 for AGZ, 100 for AZ

# Maximum number of training steps (0 continue forever)
MAX_TRAINING_STEPS = 16000 #16000 for AGZ, 500 for AZ

#!/usr/bin/env python3

# Board size and number of squares in a board. Board size must be an odd number!
BOARD_SIZE = 7
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE

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
TRAINING_STEPS = 500 #2000

# Display intermediate output after the specified number of training steps
TRAINING_STEPS_VERBOSE = 100

# Maximum number of training steps (0 continue forever)
MAX_TRAINING_STEPS = 500 #16000

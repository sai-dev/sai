#!/usr/bin/env python3

# Board size and number of squares in a board. Board size must be an odd number!
BOARD_SIZE = 7
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE

# Network structure -- common part
RESIDUAL_FILTERS = 128
RESIDUAL_BLOCKS = 3
POLICY_OUTPUTS = 2

# Network structure -- Sai value head
# Value head type can be:
SINGLE = 1 # (Leela Zero)
DOUBLE_V = 2
DOUBLE_Y = 3
DOUBLE_T = 4
DOUBLE_I = 5

VALUE_HEAD_TYPE = DOUBLE_Y
VAL_OUTPUTS = 2
VBE_OUTPUTS = 1 # only for double W
VAL_CHANS = 256
VBE_CHANS = 128 # only for double W and Y

# Learning rate
LEARN_RATE = 0.005

# Outputs new network after the specified number of training steps
TRAINING_STEPS = 2000

# Display intermediate output after the specified number of training steps
INFO_STEPS = 400

# Maximum number of training steps (0 continue forever)
MAX_TRAINING_STEPS = 2000

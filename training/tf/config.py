#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2018 SAI Team
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

# 16 planes, 1 side to move, 1 x BOARD_SQUARES probs, 1 winner = 19 lines
DATA_ITEM_LINES = 16 + 1 + 1 + 1

# Sane values are from 4096 to 64 or so.
# You need to adjust the learning rate if you change this. Should be
# a multiple of RAM_BATCH_SIZE. NB: It's rare that large batch sizes are
# actually required.
BATCH_SIZE = 512
# Number of examples in a GPU batch. Higher values are more efficient.
# The maximum depends on the amount of RAM in your GPU and the network size.
# Must be smaller than BATCH_SIZE.
RAM_BATCH_SIZE = 64

# Use a random sample input data read. This helps improve the spread of
# games in the shuffle buffer.
DOWN_SAMPLE = 16

# -- 2.2GB of RAM.
TRAIN_SHUFFLE_BITS=20
TEST_SHUFFLE_BITS=17  # was 19

# Board size and number of squares in a board. Board size must be an odd number!
BOARD_SIZE = 7
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE

# Network structure -- common part

RESIDUAL_FILTERS = 128
RESIDUAL_BLOCKS = 3
POLICY_OUTPUTS = 2
INPUT_STM = 0 # 1: both side to move and komi in input (18 input planes)
              # 0: only komi in input (17 input planes)

# Network structure -- Sai value head
# Value head type can be:
SINGLE = 1 # (Leela Zero)
DOUBLE_V = 2
DOUBLE_Y = 3
DOUBLE_T = 4 # last two tyoes are equivalent, changing
DOUBLE_I = 5 # only the order of weights in the file

VALUE_HEAD_TYPE = DOUBLE_Y
VAL_OUTPUTS = 2
VBE_OUTPUTS = 1 # only for double W
VAL_CHANS = 256
VBE_CHANS = 128 # only for double W and Y

# Learning rate
LEARN_RATE = 0.0005

# Outputs new network after the specified number of training steps
TRAINING_STEPS = 4000

# Display intermediate output after the specified number of training steps
INFO_STEPS = 800

# Maximum number of training steps (0 continue forever)
MAX_TRAINING_STEPS = 4000

#!/usr/bin/env python3
#
#    This file is part of SAI, which is a fork of Leela Zero.
#    Copyright (C) 2018-2019 SAI Team
#
#    SAI is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SAI is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with SAI.  If not, see <http://www.gnu.org/licenses/>.

# 16 planes, 1 side to move, 1 x BOARD_SQUARES probs, 1 winner = 19 lines
INPUT_PLANES = 12
DATA_ITEM_LINES = 15

# Board size and number of squares in a board. Board size must be an odd number!
BOARD_SIZE = 19
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE

# Workers
WORKERS = 8

# Sane values are from 4096 to 64 or so.
# You need to adjust the learning rate if you change this. Should be
# a multiple of RAM_BATCH_SIZE. NB: It's rare that large batch sizes are
# actually required.
BATCH_SIZE = 512
# Number of examples in a GPU batch. Higher values are more efficient.
# The maximum depends on the amount of RAM in your GPU and the network size.
# Must be smaller than BATCH_SIZE.
RAM_BATCH_SIZE = 128

# Memory allocation
GPU_MEM_FRACTION = 0.8

# 20 bit should be about 2.2GB of RAM on 19x19 and 0.5GB on 9x9
# Formula is M*S*(16/8+4) with M the shuffle buffer size, S the board
# squares, 16 the history bit fields and (1/8, 4) the size of (bit,
# float).
# Sensible number is log2(N) with N actual number of training
# positions i.e. about BOARD_SQUARES*N_GAMES. In this way, since the
# dataset is augmented eightfold with symmetries, the actual density
# of positions is about 1/8.
# In the case of "moving window" training set it is also reasonable to
# reduce more and up to a factor equal to the number of times each chunk
# enters a training.
TRAIN_SHUFFLE_BITS=21
TEST_SHUFFLE_BITS=17
# Use a random sample input data read. This helps improve the spread of
# games in the shuffle buffer.
# This should be between 2 and 4 times the ratio of N to M.
DOWN_SAMPLE = 16

# Network structure -- common part

RESIDUAL_FILTERS = 256
RESIDUAL_BLOCKS = 12
POLICY_OUTPUTS = 2
INPUT_STM = 0 # 1: both side to move and komi in input (18 input planes)
              # 0: only komi in input (17 input planes)
WEIGHTS_FILE_VER = "209"  # 'advanced features' + 'komi policy'
                         # bit 0,   1: LZ must be on
                         # bit 4,  16: advanced features (+2 planes)
                         # bit 5,  32: komi policy
                         # bit 6,  64: chain liberties features (+4 planes)
                         # bit 7, 128: chain size features (+4 planes)
KOMI_POLICY_CHANS = 14 # only used for komi policy net format

# Network structure -- Sai value head
# Value head type can be:
SINGLE = 1 # (Leela Zero)
DOUBLE_V = 2
DOUBLE_Y = 3
DOUBLE_T = 4 # last two types are equivalent, changing
DOUBLE_I = 5 # only the order of weights in the file

VALUE_HEAD_TYPE = DOUBLE_Y
VAL_OUTPUTS = 5
VBE_OUTPUTS = 1 # only for double W
VAL_CHANS = 384
VBE_CHANS = 256 # only for double W and Y

# Loss weights
POLICY_LOSS_WT = 1.0
MSE_LOSS_WT = 1.0
KLE_LOSS_WT = 0.0
AXB_LOSS_WT = 0.0
REG_LOSS_WT = 1.0

# Learning rate
LEARN_RATE = 0.001

# Outputs new network after the specified number of training steps
TRAINING_STEPS = 1000

# Display intermediate output after the specified number of training steps
INFO_STEPS = 100

# Maximum number of training steps (0 continue forever)
MAX_TRAINING_STEPS = 8000
FIRST_NETWORK_STEPS = 4000

# Maximum number of networks of which to keep meta files
MAX_SAVER_TO_KEEP = 12

# Beta value amplification coefficient
# (bugfix with backward compatibility concerns)

# If the factor is 1, then the nets will correctly estimate
# beta. Since current versions of SAI do not allow to apply a
# correction factor inside the main program, we allow the networks to
# be trained with an inherent factor. This factor was 2.0 from the
# beginning because of a bug.
BETA_SCALE_FACTOR = 2.0

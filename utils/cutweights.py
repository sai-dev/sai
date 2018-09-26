#!/bin/python3

import sys

gobansize = 7
residual_layers = 2
filters = 64

def cutline(size):
    line = sys.stdin.readline().split()
    print (*line[:size], sep=" ")

def passline():
    print (sys.stdin.readline(), end="")

def skipline():
    sys.stdin.readline()

passline()
cutline(18*9*filters)
for _ in range(3):
    cutline(filters)
for _ in range(2*residual_layers):
    cutline(filters*9*filters)
    for _ in range(3):
        cutline(filters)

for _ in range((6-residual_layers)*8):
    skipline()

cutline(filters*2)
for _ in range(3):
    passline()

cutline(2*gobansize*gobansize*(gobansize*gobansize+1))
cutline(gobansize*gobansize+1)
cutline(filters)
for _ in range(3):
    passline()
cutline(gobansize*gobansize*256)
for i in range(3):
    passline()

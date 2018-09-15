#!/bin/python3

import sys

line = sys.stdin.readline()
i = 1
while line != "":
    print (i, len(line.split()))
    i += 1
    line = sys.stdin.readline()

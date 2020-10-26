#!/bin/python3
import sys
import os
import gzip

# This script reads an LZ or SAI network file and
# writes separate csv files for the heads layers.
# These files can then be easily analyzed.

MIN_BLOCKS = 1

def skiplines(netfile, n):
    for _ in range(n):
        _ = netfile.readline()



if __name__ == '__main__':
    argc = len(sys.argv)
    if not(argc == 2):
        print('''Syntax:\n'''
              '''wts2mat.py <network.gz> ''')
        exit(1)

    gzfilename = sys.argv[1]
    if not os.path.exists(gzfilename):
        print(f"File {gzfilename} does not exists.")
        exit(1)
    netfile = gzip.open(gzfilename, mode='rt')
    hash = os.path.splitext(os.path.basename(gzfilename))[0]
    hash = hash[:8]

    print(f"Network version {netfile.readline()}", end='')
    skiplines(netfile, 1) # scrap first conv weights
    filters = len(netfile.readline().split())
    print(f"Filters {filters}")
    # scrap input conv (2 more), 6 resconv blocks (48)
    convnum = 2 * MIN_BLOCKS
    skiplines(netfile, 2 + 4 * convnum)

    w = netfile.readline().split()
    while len(w) == filters * filters * 9:
        skiplines(netfile, 3)
        w = netfile.readline().split()
        convnum += 1
    blocks = int(convnum // 2)
    assert(convnum == 2 * blocks)
    print(f"Blocks {blocks}")
    pol_filts = int(len(w) // filters)
    assert(len(w) == pol_filts * filters)
    print(f"Policy filters {pol_filts}")

    of = open(hash + '-pol-conv.csv','w')
    for infilter in range(filters):
        print(*w[infilter::filters], sep=',', file=of)
    print(file=of)

    for _ in range(3):
        w = netfile.readline().split()
        print(*w, sep=',', file=of)
    of.close()

    w = netfile.readline().split()
    assert(len(w) == pol_filts * 361 * 362)
    b = netfile.readline().split()

    of = open(hash + '-pol-dense.csv','w')
    for channel in range(362):
        for line in range(19):
            for infilter in range(2):
                ind = pol_filts * 361 * channel + 361 * infilter + 19 * line
                print(*w[ind:ind+19], sep=',', end=',,', file=of)
            if line < 18:
                print('', file=of)
            else:
                print(b[channel], file=of)
        print('', file=of)
    for line in range(19):
        print(*b[19*line::19], sep=',', file=of)
    print(b[-1], file=of)
    of.close()
    w = netfile.readline().split()
    val_filts = int(len(w) // filters)
    assert(len(w) == val_filts * filters)
    print(f"Value filters {val_filts}")

    of = open(hash + '-val-conv.csv','w')
    for infilter in range(filters):
        print(*w[infilter::filters], sep=',', file=of)
    print(file=of)

    for _ in range(3):
        w = netfile.readline().split()
        print(*w,sep=',', file=of)
    of.close()

    w = netfile.readline().split()
    b = netfile.readline().split()
    val_chans = len(b)
    assert(len(w) == val_filts * 361 * val_chans)
#    print(f"Alpha channels {val_chans}")
    w1 = netfile.readline().split()
    b1 = netfile.readline().split()

    of = open(hash + '-val-dense.csv','w')
    for channel in range(val_chans):
        for line in range(19):
            for infilter in range(val_filts):
                ind = val_filts * 361 * channel + 361 * infilter + 19 * line
                print(*w[ind:ind+19], sep=',', end=',,', file=of)
            if line < 18:
                print('', file=of)
            else:
                print(b[channel], file=of)
        print('', file=of)
    for ind in range(val_chans):
        print(b[ind], w1[ind], sep=',,', file=of)
    print('\n', b1[0], file=of)
    of.close()

    w = netfile.readline().split()
    if (len(w) == 0):
        print('''LZ network: single value head\n'''
        f'''Value channels {val_chans}''')
        exit()
    else:
        print('''SAI network: double value head\n'''
        f'''Alpha channels {val_chans}''')
        
    b = netfile.readline().split()
    vbe_chans = len(b)
    assert(len(w) == val_filts * 361 * vbe_chans)
    print(f"Beta channels {vbe_chans}")
    w1 = netfile.readline().split()
    b1 = netfile.readline().split()

    of = open(hash + '-vbe-dense.csv','w')
    for channel in range(vbe_chans):
        for line in range(19):
            for infilter in range(val_filts):
                ind = val_filts * 361 * channel + 361 * infilter + 19 * line
                print(*w[ind:ind+19], sep=',', end=',,', file=of)
            if line < 18:
                print('', file=of)
            else:
                print(b[channel], file=of)
        print('', file=of)
    for ind in range(vbe_chans):
        print(b[ind], w1[ind], sep=',,', file=of)
    print('\n', b1[0], file=of)
    of.close()

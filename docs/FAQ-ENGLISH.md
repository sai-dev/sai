# FAQ

You will find below most common questions about SAI and their answer.

## What are the main differences between SAI an Leela Zero

explain SAI vs Leela Zero and advantages of SAI

.
.
.
.
.
.
.

So speaking of Leela zero or the SAI engine are used alternatively here in a similar way.

## Why doesn't the network get stronger every time

AZ also had this behavior, besides we're testing our approach right now. Please be patient.

## Why only dozens of games are played when comparing two networks

We use SPRT to decide if a newly trained network is better. A better network is only chosen if SPRT finds it's 95% confident that the new network has a 55% (boils down to 35 elo) win rate over the previous best network.

## Why the game generated during self-play contains quite a few bad moves

The MCTS playouts of self-play games is only 3200, and with noise added (For randomness of each move thus training has something to learn from). If you load Leela Zero with Sabaki, you'll probably find it is actually not that weak.

## Very short self-play games ends with White win

This is expected.

Due to randomness of self-play games, once Black choose to pass at the beginning, there is a big chance for White to pass too (7.5 komi advantage for White). See issue [#198](https://github.com/leela-zero/leela-zero/issues/198) for details.

## Wrong score

Leela Zero uses Tromp-Taylor rules (see [here](https://senseis.xmp.net/?TrompTaylorRules).

Although its komi is 7.5 as in Chinese rule, for simplicity, Tromp-Taylor rules do not remove dead stones.

Thus, the result may be different from that calcuated using Chinese rule. However, keeping dead stones does not affect training results because both players are expected to capture dead stones themselves.

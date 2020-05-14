# autogtp

This is a self-play tool for SAI. When launched, it will fetch the
best network from the server so far, play a game against itself, and upload
the SGF and training data at the end of the game.

## Requirements

* Qt 5.3 or later with qmake
* C++14 capable compiler
* curl
* gzip and gunzip

## Matches information

Autogtp will automatically download better networks once found.

While autogtp is running, typing q+Enter will save the processed data and exit.
 When autogtp runs next, autogtp will continue the game.

Not each trained network will be a strength improvement over the prior one.
 Patience please. :)

Match games are played at full strength.

Self-play games are played with some randomness in the first moves,
 and noise all game long.

Training data from self-play games are full strength even if plays appear weak.

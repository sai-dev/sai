# What

SAI is a variable-komi fork of leela-zero, a Go program with no human
 provided knowledge.

SAI is derived from Leela Zero 0.17 + AutoGTP v18.

This means even though SAI and Leela Zero share many similarities, SAI
 brings its own specific features and things as compared to Leela Zero.

The relevant papers are:

- [SAI7x7](https://arxiv.org/abs/1809.03928)
- [SAI9x9](https://arxiv.org/abs/1905.10863)

## Current Run

The server is currently running [here](http://sai.unich.it/), on
 resources of Chieti-Pescara University.

The 19x19 run started just recently and its play is still quite random.

## Previous Runs

We have pretty strong 9x9 networks such as
 [S1](http://sai.unich.it/networks/94619dea457de054503cec030269ce842c47055ba51e96db8fee841dfbaf05f9.gz)
 from the 9x9 paper.

But you will need to compile the program with modified settings, for it
 to be able to play on 9x9 goban.

## What you can do with SAI

With SAI you can:

- run SAI locally on your machine to play with various boardsizes
 (9x9 is superhuman, 19x19 will be quite weak for many months)
- participate in the collective contributing effort to help training SAI.

Among the main differences between SAI and Leela Zero:

- support for any komi
- trained to support natively high handicap stones
- hardcoded options to make play more reliable: ladder planes, etc.

You can see a more detailed summary of the differences between SAI and Leela Zero
 [in the wiki](https://github.com/sai-dev/sai/wiki), or refer to
 [the papers](/README.md#what) for technical exhaustiveness.

## What you need

A PC with a computing device:

- ideally a GPU (a discrete graphics card made by NVIDIA or AMD,
 preferably not too old, and with the most recent drivers installed).
- or you can run the program without a GPU (CPU-only), but performance
 will be much lower. If your CPU is not recent (Haswell or newer, Ryzen
 or newer), performance will most likely be very bad, but you can still play.

Note that you can use either your own physical hardware, or run SAI
 remotely from cloud virtual machines, or another source, which can
 be a workaround to you not having your own performant computing device.

SAI can run on Windows, MacOS, and Linux.

## How to download, install, and run SAI

These steps will allow you to be able to run SAI and play with SAI:

- for Windows, see [here](/docs/RUN-WINDOWS.md)
- for Linux, see [here](/docs/RUN-LINUX.md)

## How to help SAI get stronger

If you want to help, you can use your computing device in the
 collective effort to make SAI stronger.

This will make SAI play against itself (selfplay games) and other versions
 of itself (match games) on the [SAI server](http://sai.unich.it/),
 to train and get stronger.

Any help is greatly appreciated!

### username and password

Unlike Leela Zero, to help the collective effort with games and matches,
 you have to create credentials [here](http://sai.unich.it/user-request).

This way we can track and contact people that, for any reason, are
 uploading wrong data. (This happened sometimes for leela-zero project.)

Also, this gives you access to the [The Leaderboard](http://sai.unich.it/leaderboard) !

Choose a password of no importance and that you don't use elsewhere,
 since it will be sent unencrypted.

### autogtp

To contribute, we don't run the sai executable but instead a specific
 contributing executable called autogtp.

After you downloaded and installed SAI as explained
 [previously](/README.md#how-to-download-install-and-run-sai),
 see the autogtp instructions:

- for Windows: [here](/docs/CONTRIBUTING-WINDOWS.md)
- for Linux: [here](/docs/CONTRIBUTING-LINUX.md)

The client autogtp will connect to the server automatically and do its
work in the background, uploading results after each game. You can
stop it with Ctrl-C.

Note that specific autogtp documentation is also available
 [here](/autogtp/README.md).

## How to contribute to SAI's github

You can find some guidelines on how to contribute to this github
 [here](/docs/CONTRIBUTING-GITHUB.md).

Note that these are mostly borrowed from Leela Zero's github, so
 they are not totally relevant for SAI.

## FAQ

You can find commonly asked questions about SAI and their answers
 in several languages:

- in english, see [here](/docs/FAQ-ENGLISH.md)
- in chinese 中文, see [here](/docs/FAQ-CHINESE.md)

## License

The code is released under the GPLv3 or later, except for
 ThreadPool.h, cl2.hpp, half.hpp and the eigen and clblast_level3
 subdirs, which have specific licenses (compatible with GPLv3)
 mentioned in those files.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or
 combining it with NVIDIA Corporation's libraries from the
 NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
 Network library and/or the NVIDIA TensorRT inference library
 (or a modified version of those libraries), containing parts covered
 by the terms of the respective license agreement, the licensors of
 this Program grant you additional permission to convey the resulting
 work.

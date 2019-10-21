# What

SAI is a variable-komi fork of leela-zero, a Go program with no human
provided knowledge.

The relevant papers are [SAI7x7](https://arxiv.org/abs/1809.03928) and
[SAI9x9](https://arxiv.org/abs/1905.10863).

The server is currently running [here](http://sai.unich.it/), on
resources of Chieti-Pescara University.

The 19x19 run started just recently and its play is still quite
random, but we have pretty strong 9x9 networks such as
[S1](http://sai.unich.it/networks/94619dea457de054503cec030269ce842c47055ba51e96db8fee841dfbaf05f9.gz)
from the 9x9 paper. (But you will need to compile the program with
modified settings, for it to be able to play on 9x9 goban.)

# I want to help

## Using your own hardware

You need a PC with a GPU, i.e. a discrete graphics card made by NVIDIA or AMD,
preferably not too old, and with the most recent drivers installed.

It is possible to run the program without a GPU, but performance will be much
lower. If your CPU is not *very* recent (Haswell or newer, Ryzen or newer),
performance will be outright bad, and it's probably of no use trying to join
the distributed effort. But you can still play, especially if you are patient.

### Windows

Head to the Github releases page at
https://github.com/sai-dev/sai/releases, download the latest release
and unzip. (There is also a cpu-only version if you need it.)

Then you can use the main program SAI.
You have to open a Windows command prompt in the directory with
the program to run it. It will need a network to work and networks can
be found on the [server](http://sai.unich.it/). But you can immediately launch
```
sai.exe --help
```
to see the options.

If you'd rather play with a strong SAI 9x9 network, we provide a
compiled 9x9 executable sai9x9.exe but you can also simply run
sai9x9.bat and you will play agains the
[S1](http://sai.unich.it/networks/94619dea457de054503cec030269ce842c47055ba51e96db8fee841dfbaf05f9.gz)
network from the 9x9 paper, which is included inside the zip file.

To help the collective effort with games and matches, you have to
create credentials [here](http://sai.unich.it/user-request). This way
we can track and contact people that, for any reason, are uploading
wrong data. (This happened sometimes for leela-zero project.)

Choose a password of no importance and that you don't use elsewhere,
since it will be stored unencrypted in your local machine.

Then simply double-click on sai.hta – authorization for this operation
may be requested, if so please grant it.
The possible options can be found calling from the command prompt
```
autogtp.exe --help
```
We suggest to use the option -g2 (or -g3, -g4 or larger) if your
computer is powerful enough to play more than one game at the same
time.

Please note that sai.hta is simply a wrapper for autogtp.exe and launches
```
autogtp.exe --url http://sai.unich.it/ --username <your_computername> --password <your_password> <your_options>
```

The client autogtp will connect to the server automatically and do its
work in the background, uploading results after each game. You can
just close the autogtp window to stop it.

### macOS and Linux

Follow the instructions given on leela-zero
[github](https://github.com/leela-zero/leela-zero) to compile the
sai and autogtp binaries in the build subdirectory.

Then you can use the main program SAI.
You have to open a shell and to the directory with the
program to run it. It will need a network to work and networks can be
found on the [server](http://sai.unich.it/). But you can immediately
launch
```
sai --help
```
to see the options.

If you'd rather play with a strong SAI 9x9 network, you can compile a
SAI 9x9 executable by editing src/config.cpp. A pretty strong 9x9
network is
[S1](http://sai.unich.it/networks/94619dea457de054503cec030269ce842c47055ba51e96db8fee841dfbaf05f9.gz)
from the 9x9 paper, downloadable from the link.

To help the collective effort with games and matches, you have to
create credentials [here](http://sai.unich.it/user-request). This way
we can track and contact people that, for any reason, are uploading
wrong data. (This happened sometimes for leela-zero project.)

Choose a password of no importance and that you don't use elsewhere,
since it will be sent unencrypted.

Then, be sure that sai and autogtp executables are in the $PATH and simply launch
```
autogtp --url http://sai.unich.it/ --username <your_username> --password <your_password> -g <n>
```
where n is the number of games you want to play at the same time
(depending on your hardware).

The client autogtp will connect to the server automatically and do its
work in the background, uploading results after each game. You can
stop it with Ctrl-C.

# License

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

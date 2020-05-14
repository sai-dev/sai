# CONTRIBUTING ON LINUX

macOS, Ubuntu, and more generally Linux share a very similar configuration for SAI,
 with minor differences (for example macOS uses `brew` not `apt`).

These steps show the example of Debian-based linux distributions (Ubuntu and similar).

If you're on macOS, or another type of Linux distributions (such as Fedora or Arch
 Linux, for example), you may either slightly adapt the below instructions to your
 system, or refer to the Leela Zero's original compiling instructions
 [here](https://github.com/leela-zero/leela-zero#compiling-autogtp-andor-leela-zero).

## Compile autogtp

```Shell
# Install qt5 dependencies && \
sudo apt install -y qt5-default qt5-qmake curl && \
# Compile autogtp with qmake
cd ~/sai/autogtp && \
qmake -qt5 && \
make && \
# Copy sai binary in build subdirectory in the same directory as autogtp binary && \
cp ../build/sai .
```

## Run autogtp

Then, you can run AutoGTP to start contributing:

```Shell
./sai/autogtp/autogtp --username <yourUsername> --password <yourPassword> -g <n>
```

The `-g` argument is optional, you can use it to specify the number of games (n)
 you want to play at the same time (depending on your hardware).

We suggest to use the option `-g2` (or `-g3`, `-g4` or larger) if your computer
 is powerful enough to play more than one game at the same time.

for example (default is one game at a time):

```Shell
./sai/autogtp/autogtp --username testTutorial --password 123456
```

The client autogtp will connect to the server automatically and do its
work in the background, uploading results after each game. You can
stop it with Ctrl-C.

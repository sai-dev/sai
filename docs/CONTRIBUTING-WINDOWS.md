# CONTRIBUTING ON WINDOWS

Either use your already downloaded pre-compiled SAI release, or
 if you're an experienced user, compile your own version of autogtp.

Here we cover only downloaded ready to use SAI official binaries, if you
 want to compile autogtp by yourself, see
 [here](https://github.com/leela-zero/leela-zero/tree/next/autogtp#compiling-under-visual-studio---windows).

## Setup contributing

In the SAI latest release you downloaded earlier, double-click on sai.hta.

Authorization for this operation may be requested, if so please grant it.

## Run autogtp

The possible options can be found calling from the command prompt,
 from your SAI's folder:

```Shell
autogtp.exe --help
```

To start contributing, run:

```Shell
autogtp.exe --username <yourUsername> --password <yourPassword> -g <n>
```

The `-g` argument is optional, you can use it to specify the number of games (n)
 you want to play at the same time (depending on your hardware).

We suggest to use the option `-g2` (or `-g3`, `-g4` or larger) if your computer
 is powerful enough to play more than one game at the same time.

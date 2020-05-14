# RUN SAI ON LINUX

macOS, Ubuntu, and more generally Linux share a very similar configuration for SAI,
 with minor differences (for example macOS uses `brew` not `apt`).

These steps show the example of Debian-based linux distributions (Ubuntu and similar).

If you're on macOS, or another type of Linux distributions (such as Fedora or Arch
 Linux, for example), you may either slightly adapt the below instructions to your
 system, or refer to the Leela Zero's original compiling instructions
 [here](https://github.com/leela-zero/leela-zero#compiling-autogtp-andor-leela-zero).

## Compile SAI

### 19x19

To compile the SAI binary in the build subdirectory, first test if your device is OpenCL compatible. To do that, open a shell and run:

```Shell
sudo apt install clinfo && clinfo
```

If all is good, your device is compatible with SAI, so you can run this all-in-one command to download and compile SAI:

```Shell
# Clone github repo && \
cd ~ && \
git clone https://github.com//sai && cd sai && \
git submodule update --init --recursive && \
# Install build depedencies && \
sudo apt install -y cmake g++ libboost-dev libboost-program-options-dev libboost-filesystem-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev && \
# Use a stand alone build directory to keep source dir clean && \
mkdir build && cd build && \
# Compile sai in build subdirectory with cmake && \
cmake .. && \
cmake --build .
```

Sai is installed in `~/sai`, and the sai binary is compiled and ready
 to use in `~/sai/build/`

Optionally, you can test if your build works correctly with:

```Shell
~/sai/tests
```

### 9x9

If you'd rather play with a strong SAI 9x9 network, you can compile a
SAI 9x9 executable by editing src/config.cpp.

A pretty strong 9x9 network is
 [S1](http://sai.unich.it/networks/94619dea457de054503cec030269ce842c47055ba51e96db8fee841dfbaf05f9.gz) from the 9x9 paper, downloadable from the link.

The steps to compile SAI are the same as above.

### Run SAI

Then you can use the main program SAI.

You have to open a shell and go to the directory with the
program to run it.

It will need a network to work and networks can be
found on the [server](http://sai.unich.it/). But you can immediately
launch, to see the options:

```Shell
~/sai/build/sai --help
```

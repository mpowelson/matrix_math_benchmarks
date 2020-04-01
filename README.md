# Matrix Math Benchmarks

Provides some benchmarks for matrix math libraries.

## Libraries
1) Eigen
2) PyTorch
3) ArrayFire

## Benchmarks
1) Dense Chained Multiplication - Multiplies 6 dense matrices together in a chain
2) Dense Inversion - Inverts a dense matrix
3) Dense Multiplication - Multiplies 2 dense matrices together

Each benchmark is evaluated on square matrices with increasing matrix edge size

## Install Instructions
This package relies on several matrix math libraries as well as Google Benchmark. If you are on Ubuntu install these packages as follows

### Google Benchmark
Install with the debian:

```
sudo apt install libbenchmark-dev
```

### Eigen
Install with the debian:

```
sudo apt install libeigen3-dev
```

### PyTorch

1) Download the appropriate zip file from the PyTorch website: https://pytorch.org/get-started/locally/

2) Unzip the file and navigate to it in the terminal

3) Copy it into your install directory: sudo cp -r libtorch/ /usr/local/libtorch/

4) Add libtorch to your PATH and LD_LIBRARY_PATH in your ~/.bashrc file

```
export PATH="/usr/local/libtorch/share/cmake:$PATH"
export LD_LIBRARY_PATH=/usr/local/libtorch/lib:$LD_LIBRARY_PATH
```

### ArrayFire
For best results, follow the instructions on [ArrayFire's website](http://arrayfire.org/docs/installing.htm). For subpar results, follow my instructions

1) Download the installer from ArrayFire's website: http://arrayfire.org/docs/installing.htm

2) Install into your /opt directory: `./Arrayfire_*_Linux_x86_64.sh --include-subdir --prefix=/opt`

3) Add ArrayFire to your PATH and LD_LIBRARY_PATH in your ~/.bashrc file

```
export PATH="/opt/arrayfire/share/ArrayFire/cmake:$PATH"
export LD_LIBRARY_PATH=/opt/arrayfire/lib64:$LD_LIBRARY_PATH
```

## Disclaimers
Absolutely no guarantee of fitness or optimality in these benchmarks. Use at your own risk.

Note: There are a few points of possible error here.
1) GPU calculations - I'm not sure that these benchmarks are accurate for calculations taking place on the GPU. In particular, I know that PyTorch states that GPU calculations happen asynchronously. I have not done any work to solve this
2) Lazy Execution - ArrayFire uses lazy execution. I am using the "benchmark::DoNotOptimize" tag, but I am not certain if that is enough to force it to run. It would appear that it is, since the time tends to increase as the size increases. I would not expect that if it was simply creating a computation graph

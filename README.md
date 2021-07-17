# bmm_cuda
Block Matrix Multiplication implementation on GPU using CUDA

This repository contatins block matrix multiplication on GPU using CUDA language. The goal of this project is to speed up conventional matrix multiplication algorithms. Instead of using square block in matrices, the blocks are rectangular, which can be modified in bmm.cu file.

# Usage:

In order to run the program, first make sure you have CUDA on your system. My suggestion is to use CUDA on Linux. To check if you have installed CUDA run this in termina:
```
$ nvcc --version
```
the output should be something like this:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_May__3_19:15:13_PDT_2021
Cuda compilation tools, release 11.3, V11.3.109
Build cuda_11.3.r11.3/compiler.29920130_0

```
If you don't have CUDA first make sure if you have NVIDIA graphic card. After that, from this link check if your GPU supports CUDA and install it:

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

You can found simillar infromation about how to install on other operating systems.

Now, by using this command you can complie the programm on you GPU:

```
$ nvcc -O2 bmm_main.cu bmm.cu -o bmm
```
And for running the excutable file:

```
$ ./bmm M
```
which 'M' refers to the matrix dimenseions. In other words the matrix is (2 ^ M) * (2 ^ M) and 10 <= M <= 13


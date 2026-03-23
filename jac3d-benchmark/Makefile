# Makefile for Jacobi 3D Benchmark - Linux/WSL
CC = gcc
NVCC = nvcc
CFLAGS = -O3 -Wall -fopenmp
NVCCFLAGS = -O2 -arch=sm_89
LDFLAGS = -lm

all: jac3d_cpu jac3d_cuda verify

jac3d_cpu: jac3d_cpu.c jac3d_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

jac3d_cuda: jac3d_cuda.cu jac3d_common.h
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

verify: verify.c jac3d_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f jac3d_cpu jac3d_cuda verify read_bin *.bin *.o

test: all
	./jac3d_cpu -L 384 -verify
	./jac3d_cuda -L 384 -verify
	./verify cpu_result.bin gpu_result.bin 384

.PHONY: all clean test

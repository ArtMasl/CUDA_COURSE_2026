CC = gcc
NVCC = nvcc
CFLAGS = -O3 -Wall -fopenmp
NVCCFLAGS = -O2 -arch=sm_89
LDFLAGS = -lm

ifeq ($(OS),Windows_NT)
    RM = del /Q
    EXE = .exe
else
    RM = rm -f
    EXE =
endif

all: adi3d_cpu$(EXE) adi3d_cuda$(EXE)

adi3d_cpu$(EXE): adi3d_cpu.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

adi3d_cuda$(EXE): adi3d_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	-$(RM) adi3d_cpu$(EXE) adi3d_cuda$(EXE) *.o 2>/dev/null

test: all
	./adi3d_cpu$(EXE) -L 384
	./adi3d_cuda$(EXE) -L 384

.PHONY: all clean test

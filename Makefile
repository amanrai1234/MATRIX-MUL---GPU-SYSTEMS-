# Path to NVCC compiler, since its likely not on $PATH
NVCC=/usr/local/cuda/bin/nvcc

# Xcompiler flag passes comma separated list of flag to the
# GCC/G++ compiler. Optimization flags can be pass directly
# to NVCC. O3 enables vectorization on for-loops.
NVCC_FLAGS=-std=c++11  -Xcompiler -Wall,-fopenmp,-lrt -O3

all:
	$(NVCC) $(NVCC_FLAGS) cuda_matmul.cu -o cuda_matmul

clean:
	rm cuda_matmul

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
    #include <windows.h>
    #define strcasecmp _stricmp
#else
    #include <time.h>
    #define strcasecmp strcmp
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "KERNEL ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 65535

void init_array(double* a, int L) {
    size_t total = (size_t)L * L * L;
    for (size_t idx = 0; idx < total; idx++) {
        int k = idx % L;
        int j = (idx / L) % L;
        int i = idx / (L * L);
        
        if (k == 0 || k == L - 1 || j == 0 || j == L - 1 || i == 0 || i == L - 1) {
            a[idx] = 10.0 * i / (L - 1) + 10.0 * j / (L - 1) + 10.0 * k / (L - 1);
        } else {
            a[idx] = 0.0;
        }
    }
}

__global__ void adi_x_sweep(double* a, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = L * L * L;
    
    for (int i = idx; i < total; i += stride) {
        int k = i % L;
        int j = (i / L) % L;
        int x = i / (L * L);
        
        if (x > 0 && x < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1) {
            a[i] = (a[(x-1) * L * L + j * L + k] + 
                    a[(x+1) * L * L + j * L + k]) / 2.0;
        }
    }
}

__global__ void adi_y_sweep(double* a, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = L * L * L;
    
    for (int i = idx; i < total; i += stride) {
        int k = i % L;
        int j = (i / L) % L;
        int x = i / (L * L);
        
        if (x > 0 && x < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1) {
            a[i] = (a[x * L * L + (j-1) * L + k] + 
                    a[x * L * L + (j+1) * L + k]) / 2.0;
        }
    }
}

__global__ void adi_z_sweep_reduce(double* a, double* block_max, int L) {
    __shared__ double shared_max[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = L * L * L;
    
    shared_max[tid] = 0.0;
    
    for (int i = idx; i < total; i += stride) {
        int k = i % L;
        int j = (i / L) % L;
        int x = i / (L * L);
        
        if (x > 0 && x < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1) {
            double tmp1 = (a[x * L * L + j * L + (k-1)] + 
                          a[x * L * L + j * L + (k+1)]) / 2.0;
            double tmp2 = fabs(a[i] - tmp1);
            
            if (tmp2 > shared_max[tid]) {
                shared_max[tid] = tmp2;
            }
            
            a[i] = tmp1;
        }
    }
    
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_max[blockIdx.x] = shared_max[0];
    }
}

void print_gpu_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("=== GPU Information ===\n");
    printf("CUDA devices found: %d\n", device_count);
    
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    printf("Using device: %d\n", device);
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("=======================\n\n");
}

int main(int argc, char** argv) {
    int L = 384;
    int itmax = 10;
    double maxeps = 0.01;
    
    for (int i = 1; i < argc; i++) {
        if (strcasecmp(argv[i], "-L") == 0 && i + 1 < argc) {
            L = atoi(argv[++i]);
        } else if (strcasecmp(argv[i], "-itmax") == 0 && i + 1 < argc) {
            itmax = atoi(argv[++i]);
        } else if (strcasecmp(argv[i], "-h") == 0 || strcasecmp(argv[i], "--help") == 0) {
            printf("Usage: ./adi3d_cuda [-L size] [-itmax iterations]\n");
            return 0;
        }
    }
    
    print_gpu_info();
    
    printf("=== ADI 3D CUDA ===\n");
    printf("Grid size: %d x %d x %d\n", L, L, L);
    printf("Memory required: %.2f MB\n", (double)L * L * L * sizeof(double) / (1024 * 1024));
    printf("===================\n\n");
    
    size_t total = (size_t)L * L * L;
    size_t sz = total * sizeof(double);
    
    double* h_a = (double*)malloc(sz);
    if (!h_a) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }
    
    init_array(h_a, L);
    
    double* d_a;
    CUDA_CHECK(cudaMalloc(&d_a, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice));
    
    int num_blocks = MAX_BLOCKS;
    double* d_block_max;
    CUDA_CHECK(cudaMalloc(&d_block_max, num_blocks * sizeof(double)));
    
    double* h_block_max = (double*)malloc(num_blocks * sizeof(double));
    if (!h_block_max) {
        fprintf(stderr, "Failed to allocate host memory for block_max\n");
        return 1;
    }
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    int it;
    double eps = 0.0;
    
    for (it = 1; it <= itmax; it++) {
        adi_x_sweep<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, L);
        KERNEL_CHECK();
        
        adi_y_sweep<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, L);
        KERNEL_CHECK();
        
        adi_z_sweep_reduce<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, d_block_max, L);
        KERNEL_CHECK();
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_block_max, d_block_max, num_blocks * sizeof(double),
                              cudaMemcpyDeviceToHost));
        
        eps = 0.0;
        for (int i = 0; i < num_blocks; i++) {
            if (h_block_max[i] > eps) eps = h_block_max[i];
        }
        
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        
        if (eps < maxeps) break;
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_a, d_a, sz, cudaMemcpyDeviceToHost));
    
    printf("\n=== Results ===\n");
    printf("Size            = %4d x %4d x %4d\n", L, L, L);
    printf("Iterations      = %12d\n", it);
    printf("Time in seconds = %12.4f\n", elapsed_ms / 1000.0);
    printf("Operation type  =   double precision\n");
    printf("Performance     = %10.2f MFLOPS\n",
           (2.0 * (L-2) * (L-2) * (L-2) * it * 3) / (elapsed_ms / 1000.0 * 1e6));
    printf("===============\n");
    
    free(h_a);
    free(h_block_max);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_block_max));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}

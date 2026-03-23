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

#define MAX_THREADS_PER_BLOCK 256

__global__ void copy_and_diff_kernel(double* A, double* B, double* diffs, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= L || j >= L || k >= L) return;
    
    int idx = i * L * L + j * L + k;
    
    if (i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1) {
        double diff = fabs(B[idx] - A[idx]);
        diffs[idx] = diff;
        A[idx] = B[idx];
    } else {
        diffs[idx] = 0.0;
    }
}

__global__ void jacobi_kernel(double* A, double* B, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= L || j >= L || k >= L) return;
    
    if (i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1) {
        int idx = i * L * L + j * L + k;
        B[idx] = (A[(i-1) * L * L + j * L + k] + 
                  A[i * L * L + (j-1) * L + k] + 
                  A[i * L * L + j * L + (k-1)] + 
                  A[i * L * L + j * L + (k+1)] + 
                  A[i * L * L + (j+1) * L + k] + 
                  A[(i+1) * L * L + j * L + k]) / 6.0;
    }
}

__global__ void reduce_max_kernel(double* input, double* output, size_t n) {
    __shared__ double shared[MAX_THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared[tid] = (idx < n) ? input[idx] : 0.0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmax(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
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

size_t calculate_max_L(double memory_fraction) {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    size_t available = (size_t)((double)free_mem * memory_fraction);
    size_t max_elements = available / (3 * sizeof(double));
    size_t max_L = (size_t)pow((double)max_elements, 1.0/3.0);
    max_L = (max_L / 32) * 32;
    
    if (max_L < 64) max_L = 64;
    return max_L;
}

int main(int argc, char** argv) {
    int L = 384;
    int itmax = 100;
    double maxeps = 0.5;
    int verify_mode = 0;
    double memory_fraction = 0.85;
    
    for (int i = 1; i < argc; i++) {
        if (strcasecmp(argv[i], "-L") == 0 && i + 1 < argc) L = atoi(argv[++i]);
        else if (strcasecmp(argv[i], "-itmax") == 0 && i + 1 < argc) itmax = atoi(argv[++i]);
        else if (strcasecmp(argv[i], "-maxeps") == 0 && i + 1 < argc) maxeps = atof(argv[++i]);
        else if (strcasecmp(argv[i], "-verify") == 0) verify_mode = 1;
        else if (strcasecmp(argv[i], "-memfrac") == 0 && i + 1 < argc) memory_fraction = atof(argv[++i]);
        else if (strcasecmp(argv[i], "-h") == 0 || strcasecmp(argv[i], "--help") == 0) {
            printf("Usage: ./jac3d_cuda [-L size] [-itmax iterations] [-maxeps epsilon] [-verify] [-memfrac fraction]\n");
            return 0;
        }
    }
    
    print_gpu_info();
    
    if (L == 0) {
        L = calculate_max_L(memory_fraction);
        printf("Auto-calculated L = %d based on available GPU memory\n", L);
    }
    
    printf("=== Jacobi 3D CUDA ===\n");
    printf("Grid size: %d x %d x %d\n", L, L, L);
    printf("Memory required: %.2f MB\n", 2.0 * L * L * L * sizeof(double) / (1024 * 1024));
    printf("======================\n\n");
    
    size_t total = (size_t)L * L * L;
    size_t sz = total * sizeof(double);
    
    double *d_A, *d_B, *d_diffs;
    printf("Allocating GPU memory...\n");
    CUDA_CHECK(cudaMalloc(&d_A, sz));
    CUDA_CHECK(cudaMalloc(&d_B, sz));
    CUDA_CHECK(cudaMalloc(&d_diffs, sz));
    printf("GPU memory allocated: %.2f MB\n", 3.0 * sz / 1024.0 / 1024.0);
    
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU Memory In Use: %.2f MB / %.2f MB\n\n", 
           (total_mem - free_mem) / 1024.0 / 1024.0,
           total_mem / 1024.0 / 1024.0);
    
    double* h_A = (double*)malloc(sz);
    double* h_B = (double*)malloc(sz);
    
    for (size_t idx = 0; idx < total; idx++) h_A[idx] = 0.0;
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            for (int k = 0; k < L; k++) {
                int idx = i * L * L + j * L + k;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                    h_B[idx] = 0.0;
                else
                    h_B[idx] = 4.0 + i + j + k;
            }
    
    printf("Copying data to GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));
    printf("Data copied to GPU.\n\n");
    
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((L + blockDim.x - 1) / blockDim.x,
                 (L + blockDim.y - 1) / blockDim.y,
                 (L + blockDim.z - 1) / blockDim.z);
    
    printf("Kernel config: (%d,%d,%d) blocks x (%d,%d,%d) threads\n",
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z);
    
    int num_blocks = (total + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    if (num_blocks > 65535) num_blocks = 65535;
    
    double* d_partial_max;
    CUDA_CHECK(cudaMalloc(&d_partial_max, num_blocks * sizeof(double)));
    printf("Reduction config: %d blocks x %d threads\n\n", num_blocks, MAX_THREADS_PER_BLOCK);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    int it;
    double eps = 0.0;
    double* h_partial_max = (double*)malloc(num_blocks * sizeof(double));
    
    for (it = 1; it <= itmax; it++) {
        copy_and_diff_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_diffs, L);
        KERNEL_CHECK();
        
        jacobi_kernel<<<gridDim, blockDim>>>(d_A, d_B, L);
        KERNEL_CHECK();
        
        reduce_max_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(d_diffs, d_partial_max, total);
        KERNEL_CHECK();
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_partial_max, d_partial_max, num_blocks * sizeof(double),
                              cudaMemcpyDeviceToHost));
        
        eps = 0.0;
        for (int i = 0; i < num_blocks; i++) {
            if (h_partial_max[i] > eps) eps = h_partial_max[i];
        }
        
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        
        if (eps < maxeps) break;
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    CUDA_CHECK(cudaMemcpy(h_A, d_A, sz, cudaMemcpyDeviceToHost));
    
    printf("\n=== Results ===\n");
    printf("Size            = %4d x %4d x %4d\n", L, L, L);
    printf("Iterations      = %12d\n", it);
    printf("Time in seconds = %12.4f\n", elapsed_ms / 1000.0);
    printf("Operation type  =   floating point\n");
    printf("Performance     = %10.2f MFLOPS\n",
           (2.0 * (L-2) * (L-2) * (L-2) * it * 7) / (elapsed_ms / 1000.0 * 1e6));
    printf("===============\n");
    
    if (verify_mode) {
        FILE* f = fopen("gpu_result.bin", "wb");
        if (f) {
            fwrite(h_A, sizeof(double), total, f);
            fclose(f);
            printf("GPU result saved to gpu_result.bin\n");
        }
    }
    
    free(h_A);
    free(h_B);
    free(h_partial_max);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_diffs));
    CUDA_CHECK(cudaFree(d_partial_max));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}

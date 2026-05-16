#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>

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

#define BLOCK_X 32
#define BLOCK_Y 4
#define BLOCK_Z 4
#define IDX(i, j, k, L) ((i) * (L) * (L) + (j) * (L) + (k))

__global__ void jacobi_kernel(double* A, double* B, int L) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= 0 || j <= 0 || k <= 0 || i >= L - 1 || j >= L - 1 || k >= L - 1) return;

    int idx = IDX(i, j, k, L);
    B[idx] = (A[IDX(i-1, j, k, L)] + A[IDX(i, j-1, k, L)] + A[IDX(i, j, k-1, L)] + 
              A[IDX(i, j, k+1, L)] + A[IDX(i, j+1, k, L)] + A[IDX(i+1, j, k, L)]) / 6.0;
}

__global__ void init_kernel(double* A, double* B, int L) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= L || j >= L || k >= L) return;

    int idx = IDX(i, j, k, L);
    A[idx] = 0.0;
    B[idx] = (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1) ? 0.0 : (4.0 + i + j + k);
}

struct DiffFunctor {
    const double *A;
    const double *B;
    __host__ __device__ double operator()(int idx) const { return fabs(B[idx] - A[idx]); }
};

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
    size_t max_elements = available / (2 * sizeof(double));
    size_t max_L = (size_t)pow((double)max_elements, 1.0/3.0);
    max_L = (max_L / 32) * 32;
    
    if (max_L < 64) max_L = 64;
    return max_L;
}

int main(int argc, char** argv) {
    int L = 384;
    int itmax = 20;
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
    
    double *d_A, *d_B;
    printf("Allocating GPU memory...\n");
    CUDA_CHECK(cudaMalloc(&d_A, sz));
    CUDA_CHECK(cudaMalloc(&d_B, sz));
    printf("GPU memory allocated: %.2f MB\n", 2.0 * sz / 1024.0 / 1024.0);
    
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("GPU Memory In Use: %.2f MB / %.2f MB\n\n", 
           (total_mem - free_mem) / 1024.0 / 1024.0,
           total_mem / 1024.0 / 1024.0);
    
    dim3 blockSize(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 gridSize((L + BLOCK_X - 1) / BLOCK_X, 
                  (L + BLOCK_Y - 1) / BLOCK_Y, 
                  (L + BLOCK_Z - 1) / BLOCK_Z);
    
    printf("Initializing data on GPU...\n");
    init_kernel<<<gridSize, blockSize>>>(d_A, d_B, L);
    KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Data initialized on GPU.\n\n");
    
    printf("Kernel config: %dx%dx%d blocks, %dx%dx%d threads\n", 
           gridSize.x, gridSize.y, gridSize.z, BLOCK_X, BLOCK_Y, BLOCK_Z);
    printf("Total threads: %d, Elements: %lu\n\n", 
           gridSize.x * gridSize.y * gridSize.z * BLOCK_X * BLOCK_Y * BLOCK_Z, total);
    
    int it;
    double eps = 0.0;
    
    clock_t start_t = clock();
    
    for (it = 1; it <= itmax; it++) {
        DiffFunctor diff_functor = {d_A, d_B};
        eps = thrust::transform_reduce(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(L * L * L),
            diff_functor,
            0.0,
            thrust::maximum<double>()
        );
    
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < maxeps) break;
    
        double *temp = d_A; d_A = d_B; d_B = temp;
        jacobi_kernel<<<gridSize, blockSize>>>(d_A, d_B, L);
        KERNEL_CHECK();
    }
    
    clock_t end_t = clock();
    double elapsed_sec = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    
    double* h_A = (double*)malloc(sz);
    CUDA_CHECK(cudaMemcpy(h_A, d_A, sz, cudaMemcpyDeviceToHost));
    
    printf("\n=== Results ===\n");
    printf("Size            = %4d x %4d x %4d\n", L, L, L);
    printf("Iterations      = %12d\n", it-1);
    printf("Time in seconds = %12.4f\n", elapsed_sec);
    printf("Operation type  =   floating point\n");
    printf("Performance     = %10.2f MFLOPS\n",
           (2.0 * (L-2) * (L-2) * (L-2) * it * 7) / (elapsed_sec * 1e6));
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
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    
    return 0;
}

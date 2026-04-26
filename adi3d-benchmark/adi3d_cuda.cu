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

void init_array(double* a, int L) {
    size_t total = (size_t)L * L * L;
    for (size_t idx = 0; idx < total; idx++) {
        int k = idx % L;
        int j = (idx / L) % L;
        int i = idx / (L * L);
        if (k == 0 || k == L - 1 || j == 0 || j == L - 1 || i == 0 || i == L - 1)
            a[idx] = 10.0 * i / (L - 1) + 10.0 * j / (L - 1) + 10.0 * k / (L - 1);
        else
            a[idx] = 0.0;
    }
}

__global__ void adi_x_sweep(double* __restrict__ a, int L) {
    size_t total_lines = (size_t)(L - 2) * (L - 2);
    for (size_t line_idx = blockIdx.x * blockDim.x + threadIdx.x; line_idx < total_lines; line_idx += blockDim.x * gridDim.x) {
        int dim = L - 2;
        int j = line_idx / dim + 1;
        int k = line_idx % dim + 1;
        int base = j * L + k;
        int stride = L * L;
        for (int i = 1; i < L - 1; i++) {
            int idx = base + i * stride;
            a[idx] = (a[idx - stride] + a[idx + stride]) * 0.5;
        }
    }
}

__global__ void adi_y_sweep(double* __restrict__ a, int L) {
    size_t total_lines = (size_t)(L - 2) * (L - 2);
    for (size_t line_idx = blockIdx.x * blockDim.x + threadIdx.x; line_idx < total_lines; line_idx += blockDim.x * gridDim.x) {
        int dim = L - 2;
        int i = line_idx / dim + 1;
        int k = line_idx % dim + 1;
        int base = i * L * L + k;
        int stride = L;
        for (int j = 1; j < L - 1; j++) {
            int idx = base + j * stride;
            a[idx] = (a[idx - stride] + a[idx + stride]) * 0.5;
        }
    }
}

__global__ void adi_z_sweep_reduce(double* __restrict__ a, double* __restrict__ line_max, int L) {
    size_t total_lines = (size_t)(L - 2) * (L - 2);
    for (size_t line_idx = blockIdx.x * blockDim.x + threadIdx.x; line_idx < total_lines; line_idx += blockDim.x * gridDim.x) {
        int dim = L - 2;
        int i = line_idx / dim + 1;
        int j = line_idx % dim + 1;
        int base = i * L * L + j * L;
        double local_max = 0.0;
        for (int k = 1; k < L - 1; k++) {
            int idx = base + k;
            double tmp = (a[idx - 1] + a[idx + 1]) * 0.5;
            double diff = fabs(a[idx] - tmp);
            if (diff > local_max) local_max = diff;
            a[idx] = tmp;
        }
        line_max[line_idx] = local_max;
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
        if (strcasecmp(argv[i], "-L") == 0 && i + 1 < argc) L = atoi(argv[++i]);
        else if (strcasecmp(argv[i], "-itmax") == 0 && i + 1 < argc) itmax = atoi(argv[++i]);
        else if (strcasecmp(argv[i], "-h") == 0 || strcasecmp(argv[i], "--help") == 0) {
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
    size_t total_lines = (size_t)(L - 2) * (L - 2);
    size_t sz_lines = total_lines * sizeof(double);
    double* h_a = (double*)malloc(sz);
    if (!h_a) { fprintf(stderr, "Failed to allocate host memory\n"); return 1; }
    init_array(h_a, L);
    double* d_a;
    CUDA_CHECK(cudaMalloc(&d_a, sz));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice));
    double* d_line_max;
    CUDA_CHECK(cudaMalloc(&d_line_max, sz_lines));
    int num_blocks = (total_lines + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    double* h_line_max = (double*)malloc(sz_lines);
    if (!h_line_max) { fprintf(stderr, "Failed to allocate host memory\n"); return 1; }
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
        adi_z_sweep_reduce<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, d_line_max, L);
        KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_line_max, d_line_max, sz_lines, cudaMemcpyDeviceToHost));
        eps = 0.0;
        for (size_t i = 0; i < total_lines; i++) {
            if (h_line_max[i] > eps) eps = h_line_max[i];
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
    printf("Iterations      = %12d\n", it-1);
    printf("Time in seconds = %12.4f\n", elapsed_ms / 1000.0);
    printf("Operation type  =   double precision\n");
    printf("Performance     = %10.2f MFLOPS\n",
           (2.0 * (L-2) * (L-2) * (L-2) * it * 3) / (elapsed_ms / 1000.0 * 1e6));
    printf("===============\n");
    free(h_a);
    free(h_line_max);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_line_max));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}

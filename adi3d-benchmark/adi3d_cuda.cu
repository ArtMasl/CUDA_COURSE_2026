#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

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

#define BLOCK_SIZE 32

static inline int blocks_needed(int size, int block_size) {
    return (size + block_size - 1) / block_size;
}

__global__ void init_kernel(double* __restrict__ a, int L) {
    int i = blockIdx.z;
    int j = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i >= L || j >= L || k >= L) return;

    if (i == 0 || i == L - 1 || j == 0 || j == L - 1 || k == 0 || k == L - 1)
        a[i * L * L + j * L + k] = 10.0 * i / (L - 1) + 10.0 * j / (L - 1) + 10.0 * k / (L - 1);
    else
        a[i * L * L + j * L + k] = 0.0;
}

__global__ void transpose_xyz_to_zxy_kernel(
    double* __restrict__ dst, 
    const double* __restrict__ src, 
    int nx, int ny, int nz) 
{
    __shared__ double tile[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    int i = blockIdx.z;
    int j = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (i < nx && j < ny && k < nz) {
        tile[threadIdx.y][threadIdx.x] = src[i * ny * nz + j * nz + k];
    }
    __syncthreads();
    
    j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    k = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    if (i < nx && j < ny && k < nz) {
        dst[k * nx * ny + i * ny + j] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_zxy_to_xyz_kernel(
    double* __restrict__ dst, 
    const double* __restrict__ src, 
    int nx, int ny, int nz) 
{
    __shared__ double tile[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    int i = blockIdx.z;
    int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int k = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    
    if (i < nx && j < ny && k < nz) {
        tile[threadIdx.y][threadIdx.x] = src[k * nx * ny + i * ny + j];
    }
    __syncthreads();
    
    j = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < nx && j < ny && k < nz) {
        dst[i * ny * nz + j * nz + k] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose_xyz_to_zxy(double* dst, double* src, int nx, int ny, int nz) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(blocks_needed(nz, BLOCK_SIZE), blocks_needed(ny, BLOCK_SIZE), nx);
    transpose_xyz_to_zxy_kernel<<<grid, block>>>(dst, src, nx, ny, nz);
    KERNEL_CHECK();
}

void transpose_zxy_to_xyz(double* dst, double* src, int nx, int ny, int nz) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(blocks_needed(nz, BLOCK_SIZE), blocks_needed(ny, BLOCK_SIZE), nx);
    transpose_zxy_to_xyz_kernel<<<grid, block>>>(dst, src, nx, ny, nz);
    KERNEL_CHECK();
}

__global__ void adi_x_sweep(double* __restrict__ a, int L) {
    int j = blockIdx.y * BLOCK_SIZE + threadIdx.y + 1;
    int k = blockIdx.x * BLOCK_SIZE + threadIdx.x + 1;
    if (j >= L - 1 || k >= L - 1) return;
    
    int base = j * L + k;
    int stride = L * L;
    for (int i = 1; i < L - 1; i++) {
        int idx = base + i * stride;
        a[idx] = (a[idx - stride] + a[idx + stride]) * 0.5;
    }
}

__global__ void adi_y_sweep(double* __restrict__ a, int L) {
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y + 1;
    int k = blockIdx.x * BLOCK_SIZE + threadIdx.x + 1;
    if (i >= L - 1 || k >= L - 1) return;
    
    int base = i * L * L + k;
    int stride = L;
    for (int j = 1; j < L - 1; j++) {
        int idx = base + j * stride;
        a[idx] = (a[idx - stride] + a[idx + stride]) * 0.5;
    }
}

__global__ void adi_z_sweep_transposed(
    double* __restrict__ a, 
    double* __restrict__ line_eps, 
    int nx, int ny, int nz) 
{
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y + 1;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x + 1;
    if (i >= nx - 1 || j >= ny - 1) return;
    
    double local_max = 0.0;
    int base = i * ny + j;
    for (int k = 1; k < nz - 1; k++) {
        int idx = k * nx * ny + base;
        double tmp = (a[idx - nx * ny] + a[idx + nx * ny]) * 0.5;
        double diff = fabs(a[idx] - tmp);
        if (diff > local_max) local_max = diff;
        a[idx] = tmp;
    }
    line_eps[(i - 1) * (ny - 2) + (j - 1)] = local_max;
}

void adi_z_sweep_with_transpose(double* a, double* buf, double* line_eps, int L) {
    transpose_xyz_to_zxy(buf, a, L, L, L);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(blocks_needed(L - 2, BLOCK_SIZE), blocks_needed(L - 2, BLOCK_SIZE));
    adi_z_sweep_transposed<<<grid, block>>>(buf, line_eps, L, L, L);
    KERNEL_CHECK();
    
    transpose_zxy_to_xyz(a, buf, L, L, L);
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
    
    double* h_a = (double*)malloc(sz);
    if (!h_a) { fprintf(stderr, "Failed to allocate host memory\n"); return 1; }

    double* d_a;
    CUDA_CHECK(cudaMalloc(&d_a, sz));

    dim3 init_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 init_grid(blocks_needed(L, BLOCK_SIZE), blocks_needed(L, BLOCK_SIZE), L);
    init_kernel<<<init_grid, init_block>>>(d_a, L);
    KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    double* d_line_max;
    CUDA_CHECK(cudaMalloc(&d_line_max, total_lines * sizeof(double)));
    
    double* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, sz));
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_xy(blocks_needed(L - 2, BLOCK_SIZE), blocks_needed(L - 2, BLOCK_SIZE));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    cudaEvent_t x_start, x_stop, y_start, y_stop, z_start, z_stop;
    CUDA_CHECK(cudaEventCreate(&x_start)); CUDA_CHECK(cudaEventCreate(&x_stop));
    CUDA_CHECK(cudaEventCreate(&y_start)); CUDA_CHECK(cudaEventCreate(&y_stop));
    CUDA_CHECK(cudaEventCreate(&z_start)); CUDA_CHECK(cudaEventCreate(&z_stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    int it;
    double eps = 0.0;
    
    for (it = 1; it <= itmax; it++) {
        CUDA_CHECK(cudaEventRecord(x_start));
        adi_x_sweep<<<grid_xy, block>>>(d_a, L);
        KERNEL_CHECK();
        CUDA_CHECK(cudaEventRecord(x_stop));
        
        CUDA_CHECK(cudaEventRecord(y_start));
        adi_y_sweep<<<grid_xy, block>>>(d_a, L);
        KERNEL_CHECK();
        CUDA_CHECK(cudaEventRecord(y_stop));
        
        CUDA_CHECK(cudaEventRecord(z_start));
        adi_z_sweep_with_transpose(d_a, d_buf, d_line_max, L);
        CUDA_CHECK(cudaEventRecord(z_stop));
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        thrust::device_ptr<double> dev_ptr(d_line_max);
        eps = thrust::reduce(
            thrust::device,
            dev_ptr,
            dev_ptr + total_lines,
            0.0,
            thrust::maximum<double>()
        );
        
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < maxeps) break;
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float x_ms, y_ms, z_ms, total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventElapsedTime(&x_ms, x_start, x_stop));
    CUDA_CHECK(cudaEventElapsedTime(&y_ms, y_start, y_stop));
    CUDA_CHECK(cudaEventElapsedTime(&z_ms, z_start, z_stop));
    
    printf("\n--- Sweep Timing (last iteration) ---\n");
    printf("X-sweep: %.3f ms, Y-sweep: %.3f ms, Z-sweep: %.3f ms\n", x_ms, y_ms, z_ms);
    printf("Total time: %.3f ms\n\n", total_ms);
    
    CUDA_CHECK(cudaMemcpy(h_a, d_a, sz, cudaMemcpyDeviceToHost));
    
    printf("\n=== Results ===\n");
    printf("Size            = %4d x %4d x %4d\n", L, L, L);
    printf("Iterations      = %12d\n", it - 1);
    printf("Time in seconds = %12.4f\n", total_ms / 1000.0);
    printf("Operation type  =   double precision\n");
    printf("Performance     = %10.2f MFLOPS\n",
           (2.0 * (L-2) * (L-2) * (L-2) * it * 3) / (total_ms / 1000.0 * 1e6));
    printf("===============\n");
    
    free(h_a);
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_line_max));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(x_start)); CUDA_CHECK(cudaEventDestroy(x_stop));
    CUDA_CHECK(cudaEventDestroy(y_start)); CUDA_CHECK(cudaEventDestroy(y_stop));
    CUDA_CHECK(cudaEventDestroy(z_start)); CUDA_CHECK(cudaEventDestroy(z_stop));
    
    return 0;
}

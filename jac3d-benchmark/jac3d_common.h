#ifndef JAC3D_COMMON_H
#define JAC3D_COMMON_H

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

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define Min(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    int L;
    int itmax;
    double maxeps;
    int iterations;
    double eps;
    double time_sec;
    int verified;
} Jac3DResult;

static inline double* allocate_3d_array(size_t L) {
    size_t total = L * L * L;
    return (double*)malloc(total * sizeof(double));
}

static inline void init_arrays(double* A, double* B, size_t L) {
    size_t total = L * L * L;
    for (size_t idx = 0; idx < total; idx++) {
        A[idx] = 0.0;
    }
    
    for (size_t i = 0; i < L; i++) {
        for (size_t j = 0; j < L; j++) {
            for (size_t k = 0; k < L; k++) {
                size_t idx = i * L * L + j * L + k;
                if (i == 0 || j == 0 || k == 0 || 
                    i == L - 1 || j == L - 1 || k == L - 1) {
                    B[idx] = 0.0;
                } else {
                    B[idx] = 4.0 + (double)i + (double)j + (double)k;
                }
            }
        }
    }
}

static inline int verify_arrays(double* cpu_A, double* gpu_A, size_t L, double tolerance) {
    size_t total = L * L * L;
    double max_diff = 0.0;
    
    for (size_t idx = 0; idx < total; idx++) {
        double diff = fabs(cpu_A[idx] - gpu_A[idx]);
        if (diff > max_diff) max_diff = diff;
    }
    
    printf("Max difference between CPU and GPU: %e\n", max_diff);
    return (max_diff < tolerance) ? 1 : 0;
}

static inline double get_time() {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

static inline void save_result(const char* filename, double* data, size_t L) {
    FILE* f = fopen(filename, "wb");
    if (f) {
        fwrite(data, sizeof(double), L * L * L, f);
        fclose(f);
        printf("Result saved to %s\n", filename);
    }
}

static inline double* load_result(const char* filename, size_t L) {
    double* data = allocate_3d_array(L);
    FILE* f = fopen(filename, "rb");
    if (f) {
        fread(data, sizeof(double), L * L * L, f);
        fclose(f);
    }
    return data;
}

#endif /* JAC3D_COMMON_H */
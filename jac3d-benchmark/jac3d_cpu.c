#include "jac3d_common.h"
#include <omp.h>

void jac3d_cpu(double* A, double* B, size_t L, int itmax, double maxeps, 
               Jac3DResult* result) {
    size_t i, j, k;
    double startt = get_time();
    int it;
    double eps = 0.0;
    
    for (it = 1; it <= itmax; it++) {
        eps = 0.0;
        
        #pragma omp parallel for collapse(3) reduction(max:eps)
        for (i = 1; i < L - 1; i++) {
            for (j = 1; j < L - 1; j++) {
                for (k = 1; k < L - 1; k++) {
                    size_t idx = i * L * L + j * L + k;
                    double tmp = fabs(B[idx] - A[idx]);
                    if (tmp > eps) eps = tmp;
                    A[idx] = B[idx];
                }
            }
        }
        
        #pragma omp parallel for collapse(3)
        for (i = 1; i < L - 1; i++) {
            for (j = 1; j < L - 1; j++) {
                for (k = 1; k < L - 1; k++) {
                    size_t idx = i * L * L + j * L + k;
                    B[idx] = (A[(i-1) * L * L + j * L + k] + 
                              A[i * L * L + (j-1) * L + k] + 
                              A[i * L * L + j * L + (k-1)] + 
                              A[i * L * L + j * L + (k+1)] + 
                              A[i * L * L + (j+1) * L + k] + 
                              A[(i+1) * L * L + j * L + k]) / 6.0;
                }
            }
        }
        
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        
        if (eps < maxeps) {
            break;
        }
    }
    
    double endt = get_time();
    
    result->iterations = it;
    result->eps = eps;
    result->time_sec = endt - startt;
    result->verified = 1;
}

int main(int argc, char** argv) {
    int i;
    size_t L = 384;
    int itmax = 20;
    double maxeps = 0.5;
    int verify_mode = 0;
    
    for (i = 1; i < argc; i++) {
        if (strcasecmp(argv[i], "-L") == 0 && i + 1 < argc) {
            L = (size_t)atol(argv[++i]);
        } else if (strcasecmp(argv[i], "-itmax") == 0 && i + 1 < argc) {
            itmax = atoi(argv[++i]);
        } else if (strcasecmp(argv[i], "-maxeps") == 0 && i + 1 < argc) {
            maxeps = atof(argv[++i]);
        } else if (strcasecmp(argv[i], "-verify") == 0) {
            verify_mode = 1;
        } else if (strcasecmp(argv[i], "-h") == 0 || strcasecmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-L size] [-itmax iterations] [-maxeps epsilon] [-verify]\n", argv[0]);
            printf("  -L       : Grid size (default: 384)\n");
            printf("  -itmax   : Max iterations (default: 20)\n");
            printf("  -maxeps  : Convergence threshold (default: 0.5)\n");
            printf("  -verify  : Enable verification mode (save result)\n");
            return 0;
        }
    }
    
    printf("=== Jacobi 3D CPU (OpenMP) ===\n");
    printf("Grid size: %zu x %zu x %zu\n", L, L, L);
    printf("Memory required: %.2f MB\n", 2.0 * L * L * L * sizeof(double) / (1024 * 1024));
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    printf("================================\n\n");
    
    double* A = allocate_3d_array(L);
    double* B = allocate_3d_array(L);
    
    if (!A || !B) {
        fprintf(stderr, "Failed to allocate memory for %zu^3 grid\n", L);
        return 1;
    }
    
    init_arrays(A, B, L);
    
    Jac3DResult result = {0};
    result.L = (int)L;
    result.itmax = itmax;
    result.maxeps = maxeps;
    
    jac3d_cpu(A, B, L, itmax, maxeps, &result);
    
    printf("\n=== Results ===\n");
    printf("Size            = %4d x %4d x %4d\n", (int)L, (int)L, (int)L);
    printf("Iterations      = %12d\n", result.iterations-1);
    printf("Time in seconds = %12.4f\n", result.time_sec);
    printf("Operation type  =   floating point\n");
    printf("Performance     = %10.2f MFLOPS\n", 
           (2.0 * (L-2) * (L-2) * (L-2) * result.iterations * 7) / (result.time_sec * 1e6));
    printf("===============\n");
    
    if (verify_mode) {
        save_result("cpu_result.bin", A, L);
    }
    
    free(A);
    free(B);
    
    return 0;
}

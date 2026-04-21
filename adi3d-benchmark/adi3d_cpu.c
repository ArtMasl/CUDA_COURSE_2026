#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <time.h>
#endif

#define Max(a, b) ((a) > (b) ? (a) : (b))

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

double get_time() {
    return omp_get_wtime();
}

int main(int argc, char** argv) {
    int L = 384;
    int itmax = 10;
    double maxeps = 0.01;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-L") == 0 && i + 1 < argc) {
            L = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-itmax") == 0 && i + 1 < argc) {
            itmax = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: ./adi3d_cpu [-L size] [-itmax iterations]\n");
            return 0;
        }
    }
    
    printf("=== ADI 3D CPU (OpenMP) ===\n");
    printf("Grid size: %d x %d x %d\n", L, L, L);
    printf("Memory required: %.2f MB\n", (double)L * L * L * sizeof(double) / (1024 * 1024));
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    printf("============================\n\n");
    
    size_t total = (size_t)L * L * L;
    size_t sz = total * sizeof(double);
    
    double* a = (double*)malloc(sz);
    if (!a) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }
    
    init_array(a, L);
    
    double startt = get_time();
    
    int it;
    double eps = 0.0;
    
    for (it = 1; it <= itmax; it++) {
        eps = 0.0;
        
        #pragma omp parallel for collapse(3)
        for (int i = 1; i < L - 1; i++)
            for (int j = 1; j < L - 1; j++)
                for (int k = 1; k < L - 1; k++) {
                    int idx = i * L * L + j * L + k;
                    a[idx] = (a[(i-1) * L * L + j * L + k] + 
                              a[(i+1) * L * L + j * L + k]) / 2.0;
                }
        
        #pragma omp parallel for collapse(3)
        for (int i = 1; i < L - 1; i++)
            for (int j = 1; j < L - 1; j++)
                for (int k = 1; k < L - 1; k++) {
                    int idx = i * L * L + j * L + k;
                    a[idx] = (a[i * L * L + (j-1) * L + k] + 
                              a[i * L * L + (j+1) * L + k]) / 2.0;
                }
        
        #pragma omp parallel for collapse(3) reduction(max:eps)
        for (int i = 1; i < L - 1; i++)
            for (int j = 1; j < L - 1; j++)
                for (int k = 1; k < L - 1; k++) {
                    int idx = i * L * L + j * L + k;
                    double tmp1 = (a[i * L * L + j * L + (k-1)] + 
                                   a[i * L * L + j * L + (k+1)]) / 2.0;
                    double tmp2 = fabs(a[idx] - tmp1);
                    if (tmp2 > eps) eps = tmp2;
                    a[idx] = tmp1;
                }
        
        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        
        if (eps < maxeps) break;
    }
    
    double endt = get_time();
    
    printf("\n=== Results ===\n");
    printf("Size            = %4d x %4d x %4d\n", L, L, L);
    printf("Iterations      = %12d\n", it);
    printf("Time in seconds = %12.4f\n", endt - startt);
    printf("Operation type  =   double precision\n");
    printf("Performance     = %10.2f MFLOPS\n",
           (2.0 * (L-2) * (L-2) * (L-2) * it * 3) / ((endt - startt) * 1e6));
    printf("===============\n");
    
    free(a);
    
    return 0;
}

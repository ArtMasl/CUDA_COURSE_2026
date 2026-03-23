#include "jac3d_common.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <cpu_result.bin> <gpu_result.bin> [L]\n", argv[0]);
        return 1;
    }
    
    const char* cpu_file = argv[1];
    const char* gpu_file = argv[2];
    size_t L = (argc > 3) ? (size_t)atol(argv[3]) : 384;
    
    printf("=== Verification ===\n");
    printf("CPU file: %s\n", cpu_file);
    printf("GPU file: %s\n", gpu_file);
    printf("Grid size: %zu\n", L);
    
    double* cpu_data = allocate_3d_array(L);
    double* gpu_data = allocate_3d_array(L);
    
    FILE* f_cpu = fopen(cpu_file, "rb");
    FILE* f_gpu = fopen(gpu_file, "rb");
    
    if (!f_cpu || !f_gpu) {
        fprintf(stderr, "Failed to open result files\n");
        return 1;
    }
    
    fread(cpu_data, sizeof(double), L * L * L, f_cpu);
    fread(gpu_data, sizeof(double), L * L * L, f_gpu);
    
    fclose(f_cpu);
    fclose(f_gpu);
    
    int verified = verify_arrays(cpu_data, gpu_data, L, 1e-6);
    
    printf("Verification: %s\n", verified ? "SUCCESSFUL" : "UNSUCCESSFUL");
    printf("====================\n");
    
    free(cpu_data);
    free(gpu_data);
    
    return verified ? 0 : 1;
}
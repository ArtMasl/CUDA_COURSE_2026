#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <file.bin> [L]\n", argv[0]);
        return 1;
    }

    int i, j, k;
    int L = (argc > 2) ? atoi(argv[2]) : 384;
    FILE* f = fopen(argv[1], "rb");
    if (!f) { printf("Cannot open file\n"); return 1; }
    
    double* data = (double*)malloc(L * L * L * sizeof(double));
    fread(data, sizeof(double), L * L * L, f);
    fclose(f);
    
    printf("File: %s\n", argv[1]);
    printf("Size: %d x %d x %d\n", L, L, L);
    printf("\nFirst 10 values:\n");
    for (i = 0; i < 10; i++) printf("  [%d] = %.6f\n", i, data[i]);
    printf("\nMiddle slice (L/2, :, :):\n");
    int mid = L / 2;
    for (j = 0; j < 5; j++) {
        for (k = 0; k < 5; k++) {
            printf("%8.3f ", data[mid * L * L + j * L + k]);
        }
        printf("\n");
    }
    
    free(data);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>

// Kernel to add two vectors
__global__ 
void addVectors(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

// Function to initialize vectors with random values
void initialize(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = rand() % 10;
    }
}

// Function to print vectors
void print(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");
}

int main() {
    int N = 50;  // Size of the vectors
    int* A, * B, * C;

    size_t vectorBytes = N * sizeof(int);
    
    // Allocate memory for vectors on host
    A = (int*)malloc(vectorBytes);
    B = (int*)malloc(vectorBytes);
    C = (int*)malloc(vectorBytes);

    // Initialize vectors with random values
    initialize(A, N);
    initialize(B, N);

    printf("Vector A: ");
    print(A, N);
    printf("Vector B: ");
    print(B, N);

    // Allocate memory on the device for vectors
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, vectorBytes);
    cudaMalloc(&d_B, vectorBytes);
    cudaMalloc(&d_C, vectorBytes);

    // Copy data from host to device
    cudaMemcpy(d_A, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, vectorBytes, cudaMemcpyHostToDevice);

    // Define number of threads and blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel for vector addition
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result back to the host
    cudaMemcpy(C, d_C, vectorBytes, cudaMemcpyDeviceToHost);

    printf("Result Vector C (A + B): ");
    print(C, N);

    // Free memory for vectors on device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free memory for vectors on host
    free(A);
    free(B);
    free(C);

    return 0;
}

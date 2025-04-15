#include <stdio.h>
#include <stdlib.h>

// Kernel to multiply vector and matrix
__global__ 
void multiplyVectorMatrix(int* V, int* M, int* R, int vectorSize, int matrixColumns) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < matrixColumns) {
        int sum = 0;
        for (int i = 0; i < vectorSize; i++) {
            sum += V[i] * M[i * matrixColumns + tid];  
            // M[i * matrixColumns + tid] is the element from row i and column tid
        }
        R[tid] = sum;
    }
}

// Function to initialize vector with random values
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
    int vectorSize = 4;  // Size of the vector
    int matrixRows = vectorSize;  // Rows of matrix (same as vector size)
    int matrixColumns = 4;  // Columns of matrix (could be any number)

    int* V;  // The input vector
    int* M;  // The input matrix
    int* R;  // The resulting vector

    size_t matrixBytes = matrixRows * matrixColumns * sizeof(int);
    size_t resultBytes = matrixColumns * sizeof(int);

    // Allocate memory for host
    V = (int*)malloc(vectorSize * sizeof(int));
    M = (int*)malloc(matrixBytes);
    R = (int*)malloc(resultBytes);

    // Initialize vector and matrix with random values
    initialize(V, vectorSize);
    for (int i = 0; i < matrixRows; i++) {
        for (int j = 0; j < matrixColumns; j++) {
            M[i * matrixColumns + j] = rand() % 10;
        }
    }

    printf("Vector V: ");
    print(V, vectorSize);
    
    printf("Matrix M:\n");
    for (int i = 0; i < matrixRows; i++) {
        for (int j = 0; j < matrixColumns; j++) {
            printf("%d ", M[i * matrixColumns + j]);
        }
        printf("\n");
    }

    // Allocate memory on the device
    int* d_V, * d_M, * d_R;
    cudaMalloc(&d_V, vectorSize * sizeof(int));
    cudaMalloc(&d_M, matrixBytes);
    cudaMalloc(&d_R, resultBytes);

    // Copy data to device
    cudaMemcpy(d_V, V, vectorSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, matrixBytes, cudaMemcpyHostToDevice);

    // Launch the kernel for vector-matrix multiplication
    int threadsPerBlock = 256;
    int blocksPerGrid = (matrixColumns + threadsPerBlock - 1) / threadsPerBlock;
    multiplyVectorMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_M, d_R, vectorSize,
    matrixColumns);

    // Copy the result back to the host
    cudaMemcpy(R, d_R, resultBytes, cudaMemcpyDeviceToHost);

    printf("Resultant Vector R from vector-matrix multiplication: ");
    print(R, matrixColumns);

    // Free memory
    free(V);
    free(M);
    free(R);
    cudaFree(d_V);
    cudaFree(d_M);
    cudaFree(d_R);

    return 0;
}

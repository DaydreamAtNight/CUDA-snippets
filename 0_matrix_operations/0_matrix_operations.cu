#include <stdio.h>

template <class T>
__global__ void matrixElementAdd(const T *A, const T *B, T *C, int width)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width)
    {
        C[row*width+col] = A[row*width+col] + B[row*width+col];
    }
}

template <class T>
__global__ void matrixElementMulV1(const T *A, const T *B, T *C, int width)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    T value = 0;
    if (col < width)
    {
        for (int i = 0; i < width; i++)
        {
            value += A[row*width+i] * B[i*width+col];
        }
        C[row*width+col] = value;
    }
}

template <class T>
__global__ void matrixElementMulV2(const T *A, const T *B, T *C, int width)
{
    // Shared memory of A and B submatrices, size of tile width
    int tilewidth = 16;
    __shared__ float subA[16][16];
    __shared__ float subB[16][16];

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (col < width)
    {
        T value = 0;
        for (int m = 0; m < (width+tilewidth-1) / tilewidth; m++) 
        {
            // Bring one element from each devM and devN into shared memory
            subA[threadIdx.y][threadIdx.x] = A[row * width + (m * tilewidth + threadIdx.x)];
            subB[threadIdx.y][threadIdx.x] = B[(m * tilewidth + threadIdx.y) * width + col];
            __syncthreads();
            
            // Accumulate subset of dot product
            for (int k = 0; k < tilewidth; ++k)
            {
                value += subA[threadIdx.y][k] * subB[k][threadIdx.x]; 
            }
            __syncthreads();
            
        }
        C[row * width + col] = value;
    }
}

template <class T>
__host__ void varifyMatrixElementAdd(const T *h_A, const T *h_B, T *hcol, int width)
{
    // Verify that the result vector is correct
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            if (fabs(h_A[j*width+i] + h_B[j*width+i] - hcol[j*width+i]) > 1e-5)
            {
            fprintf(stderr, "Result verification failed at element %d, %d!\n", i, j);
            exit(EXIT_FAILURE);
            }
        }
    }

    printf("Test PASSED\n");
}

template <class T>
__host__ void varifyMatrixElementMul(const T *h_A, const T *h_B, T *hcol, int width)
{
    // Verify that the result vector is correct
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            T value = 0;
            for (int k = 0; k < width; ++k)
            {
                value += h_A[j*width+k] * h_B[k*width+i];
            }
            if (fabs(value - hcol[j*width+i]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d, %d!\n", i, j);
                fprintf(stderr, "value = %f ,\t value = %f \n", value, hcol[j*width+i]);
                // exit(EXIT_FAILURE);
            }
        }
    }

    printf("Test PASSED\n");
}

void testMtatrixElementAdd()
{
    // Print the vector length to be used, and compute its size
    int width = 3000;
    size_t size = width * width * sizeof(float);
    printf("[Matrix multiply of width %d]\n", width);

    // Allocate the host and device vectors
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *hcol = (float *)malloc(size);

    float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);
    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);
    float *dcol = NULL;
    cudaMalloc((void **)&dcol, size);

    // populate the host input vectors
    for (int i = 0; i < width*width; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    dim3 threadsPerBlock (16,16);
    dim3 blocksPerGrid ((width + 16 - 1) / 16, (width + 16 - 1) / 16);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
    matrixElementAdd<float><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, dcol, width);

    // Copy the device result vector C in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(hcol, dcol, size, cudaMemcpyDeviceToHost);

    varifyMatrixElementAdd<float>(h_A, h_B, hcol, width);

    // Free device global and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(dcol);
    free(h_A);
    free(h_B);
    free(hcol);

    printf("Done\n");
}

void testMtatrixElementMul()
{
    // Print the vector length to be used, and compute its size
    int width = 3000;
    size_t size = width * width * sizeof(double);
    printf("[Matrix multiply of width %d]\n", width);

    // Allocate the host and device vectors
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *hcol = (double *)malloc(size);

    double *d_A = NULL;
    cudaMalloc((void **)&d_A, size);
    double *d_B = NULL;
    cudaMalloc((void **)&d_B, size);
    double *dcol = NULL;
    cudaMalloc((void **)&dcol, size);

    // populate the host input vectors
    for (int i = 0; i < width*width; ++i)
    {
        h_A[i] = rand()/(double)RAND_MAX;
        h_B[i] = rand()/(double)RAND_MAX;
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    dim3 threadsPerBlock (16,16);
    dim3 blocksPerGrid ((width + 16 - 1) / 16, (width + 16 - 1) / 16);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x, threadsPerBlock.x);
    // matrixElementMulV1<double><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, dcol, width);
    matrixElementMulV2<double><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, dcol, width);


    // Copy the device result vector C in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(hcol, dcol, size, cudaMemcpyDeviceToHost);

    varifyMatrixElementMul<double>(h_A, h_B, hcol, width);

    // Free device global and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(dcol);
    free(h_A);
    free(h_B);
    free(hcol);

    printf("Done\n");
}

int main(void)
{
    // testMtatrixElementAdd();
    testMtatrixElementMul();
    return 0;
}

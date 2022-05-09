/*
    Created by: Andrew Sexton (cuda version Implemented by me(Aman Rai Saxena))
          Date: March 21nd, 2022

    CSC258/458 - Parallel & Distributed Systems.
*/
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <time.h>

/* Use this macro to catch and print out runtime errors from the GPU */
/* Ex. cudaErrChk(cudaMalloc(...)) */
/*     cudaErrChk(cudaDeviceSynchronize()) */
#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cout << "GPUAssert: " << cudaGetErrorString(code) << " " << file << " line " << line << std::endl;
        if (abort) { exit(code); }
    }
}

/* Vectorizable version of matrix multiplication for comparison */
void seq_matmul(const float* A, const float* B, float* C, int nsize) {
    float temp;
    for (int i = 0; i < nsize; i++) {
        for (int j = 0; j < nsize; j++) {
            temp = 0.0f;
            for (int k = 0; k < nsize; k++) {
                temp += A[k + (i * nsize)] * B[j + (k * nsize)];
            }
            C[j + (i * nsize)] = temp;
        }
    }
}

/* Simple OMP version of matrix multiplication for comparison */
void omp_matmul(const float* A, const float* B, float* C, int nsize) {
    # pragma omp parallel
    {
        float temp;
        # pragma omp for private(temp)
        for (int i = 0; i < nsize; i++) {
            for (int j = 0; j < nsize; j++) {
                temp = 0.0f;
                for (int k = 0; k < nsize; k++) {
                    temp += A[k + (i * nsize)] * B[j + (k * nsize)];
                }
                C[j + (i * nsize)] = temp;
            }
        }
    }
}

// Function for verifying values between two arrays
// by computing abs(x[i] - Y[i]) < EPSILON
void verify(const float* X, const float* Y, int nsize){
    float EPSILON = 1E-4;
    for(int i = 0; i < nsize; i++) {
        for(int j = 0; j < nsize; j++) {
            int idx = j + (i * nsize);

            if(std::fabs(X[idx] - Y[idx]) > EPSILON) {
                std::cout << std::setprecision(15) << "(" << i << ", " << j << "): " << X[idx] << " != " << Y[idx] << std::endl;
            }
        }
    }
}

// Print a comma-separated 2D array to stdout
void print_array(const float* arr, int nsize) {
    for(int i = 0; i < nsize; i++) {
        for(int j = 0; j < nsize; j++) {
            std::cout << arr[j + (i * nsize)];

            if(j < nsize) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
    }
}

// GPU Kernel
__global__ void gpu_matmul(float* A, float* B, float* C, int nsize) {
    /* Add your code here */
    //
    // calculate the uid block
    // calculates unique thread ID in the block







    //
    /*===================*/
    float r=blockIdx.y*blockDim.y + threadIdx.y;
    float c=blockIdx.x*blockDim.x + threadIdx.x;
    float result=0;
    if(r<m && c<k){
        for(int i=0;i<nsize;i++){
            result+=A[r*nsize+i] * B[i*nsize+c];
        }
        C[r*nsize+c]=result;

    }














}


int main(int argc, char *argv[]) {
    if(argc < 2) {
        std::cout << "Invalid number of arguments: usage " << argv[0] << " <array size>" << std::endl;
        exit(0);
    }

    // Array size
    int nsize = std::atoi(argv[1]);

    // Timing Stuff
    timespec seq_start, seq_stop;
    timespec omp_start, omp_stop;
    timespec gpu_start, gpu_stop;

    // CPU side arrays
    // Arrays are long 1D, indexing is (i, j) => j + (i * nsize)
    // this gives a single index into the array using two loop variables
    float* A = new float[nsize * nsize]();
    float* B = new float[nsize * nsize]();
    float* C = new float[nsize * nsize]();

    // Fill CPU side arrays
    for(int i = 0; i < nsize; i++) {
        for(int j = 0; j < nsize; j++) {
            int idx = float(j + (i * nsize));
            A[idx] = idx + 1.0f;
            B[idx] = 1.0f / (idx + 1.0f);
        }
    }

    // Stop GPU timer
    clock_gettime(CLOCK_REALTIME, &gpu_start);



    








//////////////////////////////////////////////////////////////////////
    /* Add your code here */
    //
    //

    // RAM memory allocation
    int *l_a, *l_b, *l_c;
    cudaMallocHost((void **)&l_a, sizeof(int) * nsize * nsize);
    cudaMallocHost((void **)&l_b, sizeof(int) * nsize * nsize);
    cudaMallocHost((void **)&l_c, sizeof(int) * nsize * nsize);

    // GPU allocation

    int *A1, *B1, *C1;
    cudaMalloc((void **)&A1, sizeof(int) * nsize * nsize);
    cudaMalloc((void **)&B1, sizeof(int) * nsize * nsize);
    cudaMalloc((void **)&C1, sizeof(int) * nsize * nsize);

    // RAM to GPU

    cudaMemcpy(A1, l_a, sizeof(int) * nsize * nsize, cudaMemcpyHostToDevice);
    cudaMemcpy(B1, l_b, sizeof(int) * nsize * nsize, cudaMemcpyHostToDevice);

    dim3 dimGrid(32, 32,1);
    dim3 dimBlock(32, 32,1);

    gpu_matmul<<<dimGrid, dimBlock>>>(A1, B1, C1, nsize);


    //GPU to RAM
    cudaMemcpy(l_c,C1, sizeof(int)*nsize*nsize, cudaMemcpyDeviceToHost);



    //
    /*=======









    for (int i = 0; i <= 3; i++)
    {
        t = clock();
        GPUmatmul<<<dim3(16, 16, 16), dim3(16, 8, 8)>>>(N, A, B, C);
        cudaDeviceSynchronize();
        t = clock() - t;
        if (i)
            avg += t; // we will ignore the first run
        printf("It took GPU-%d %f ms.\n", i, (((double)t) / CLOCKS_PER_SEC) * 1000);
    }
    avg /= 3;
    avg /= CLOCKS_PER_SEC;
    avg *= 1000;
    printf("It took %lf ms on avg.\n", avg);


    cudaFree(A);
    cudaFree(B);







    ============*/

    //////////////////////////////////////////////////////////////////////////////////
    // Stop GPU timer






















    clock_gettime(CLOCK_REALTIME, &gpu_stop);	
    std::cout << "GPU Time: " << ((gpu_stop.tv_sec - gpu_start.tv_sec) + (gpu_stop.tv_nsec - gpu_start.tv_nsec) / 1E9) << '\n';

    // Compute Vectorized version
    // Modifies C in place.
    clock_gettime(CLOCK_REALTIME, &seq_start);
    seq_matmul(A, B, C, nsize);
    clock_gettime(CLOCK_REALTIME, &seq_stop);
    std::cout << "Seq (vectorized) Time: " << ((seq_stop.tv_sec - seq_start.tv_sec) + (seq_stop.tv_nsec - seq_start.tv_nsec) / 1E9) << '\n';

    // Compute OMP version
    // Modifies C in place.
    clock_gettime(CLOCK_REALTIME, &omp_start);
    omp_matmul(A, B, C, nsize);
    clock_gettime(CLOCK_REALTIME, &omp_stop);
    std::cout << "OMP Time: " << ((omp_stop.tv_sec - omp_start.tv_sec) + (omp_stop.tv_nsec - omp_start.tv_nsec) / 1E9) << '\n';
    


    verify(C,l_c,nsize);



    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(A1);
    cudaFree(B1);
    cudaFree(C1);
    cudaFreeHost(l_a);
    cudaFreeHost(l_b);
    cudaFreeHost(l_c);

    return 0;
}

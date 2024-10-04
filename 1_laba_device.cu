#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <string>

#define VECTOR_SIZE 10000000
#define CHECK_CUDA_ERROR(a) {                                                   \
    cudaError_t err = a;                                                        \
    if (err != cudaSuccess) {                                                   \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__    \
                  << ": " << cudaGetErrorString(err) << std::endl;              \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

__global__ void sumvec_d(double* a, double* b, double *c, int N) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < N) {
        c[global_id] = sin(a[global_id])*sin(a[global_id]) + cos(b[global_id])*cos(b[global_id]);
    }
}


int main(int argc, char** argv) {
    size_t N;
    if (argc > 1)
        N = atoi(argv[1]);
    else 
        N = VECTOR_SIZE;
        
    double *h_a, *h_b, *h_c;
    double *d_a, *d_b, *d_c;

    h_a = (double*)malloc(N*sizeof(double));
    h_b = (double*)malloc(N*sizeof(double));
    h_c = (double*)malloc(N*sizeof(double));

    memset(h_c, 0, N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0;
        h_b[i] = 1.0;
    }

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemset(d_a, 0, N * sizeof(double)));

    cudaEvent_t t1, t2;
    CHECK_CUDA_ERROR(cudaEventCreate(&t1));
    CHECK_CUDA_ERROR(cudaEventCreate(&t2));

    cudaEvent_t copy_start_1, copy_stop_1;
    cudaEvent_t copy_start_2, copy_stop_2;
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_start_1));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_stop_1));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_start_2));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_stop_2));

    CHECK_CUDA_ERROR(cudaEventRecord(copy_start_1));
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_stop_1));

    int threads = 128;
    int blocks = N / threads + 1;
    CHECK_CUDA_ERROR(cudaEventRecord(t1));
    sumvec_d<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CHECK_CUDA_ERROR(cudaEventRecord(t2));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_start_1));
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, N * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_stop_2));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    float t_kernel = 0, t_copy_1 = 0, t_copy_2 = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_kernel, t1, t2));  
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_copy1, copy_start_1, copy_stop_1)); 
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_copy2, copy_start_2, copy_stop_1)); 
    std::cout << "t_kernel: " << t_kernel  << " ms" << std::endl;
    std::cout << "t_copy: " << t_copy_1 + t_copy_2 << " ms" << std::endl;

    free(h_a);
    free(h_b);
    free(h_c);

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    CHECK_CUDA_ERROR(cudaEventDestroy(t1));
    CHECK_CUDA_ERROR(cudaEventDestroy(t2));
    CHECK_CUDA_ERROR(cudaEventDestroy(copy_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(copy_stop));

    return 0;
}

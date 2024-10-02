#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>

#define VECTOR_SIZE 10000000
#define STREAM_NUM 2

#define CHECK_CUDA_ERROR(call) {                                        \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl;      \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

__global__ void sumvec_d(double* a, double* b, double *c, int N) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < N) {
        c[global_id] = sin(a[global_id])*sin(a[global_id]) + cos(b[global_id])*cos(b[global_id]);
    }
}


int main(int argc, char** argv) {
    size_t N;
    size_t numStreams; 
    if (argc > 2){
        N = atoi(argv[1]);
        numStreams = atoi(argv[2]);
    }
    else if (argc > 1){	    
        N = atoi(argv[1]);
	    numStreams = STREAM_NUM;
    }
    else {
        N = VECTOR_SIZE;
        numStreams= STREAM_NUM;
    }

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

    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    int segmentSize = N / numStreams;

    cudaEvent_t t1, t2;
    CHECK_CUDA_ERROR(cudaEventCreate(&t1));
    CHECK_CUDA_ERROR(cudaEventCreate(&t2));
    cudaEvent_t copy_start, copy_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&copy_stop));
    CHECK_CUDA_ERROR(cudaEventRecord(copy_start));
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a + i * segmentSize, h_a + i * segmentSize, segmentSize * sizeof(double), cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b + i * segmentSize, h_b + i * segmentSize, segmentSize * sizeof(double), cudaMemcpyHostToDevice, streams[i]));
    }
    CHECK_CUDA_ERROR(cudaEventRecord(copy_stop));
    int threads = 128;
    int blocks = (segmentSize + threads - 1) / threads;
    CHECK_CUDA_ERROR(cudaEventRecord(t1));
    for (int i = 0; i < numStreams; ++i) {
        sumvec_d<<<blocks, threads, 0, streams[i]>>>(d_a + i * segmentSize, d_b + i * segmentSize, d_c + i * segmentSize, segmentSize);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(t2));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c + i * segmentSize, d_c + i * segmentSize, segmentSize * sizeof(double), cudaMemcpyDeviceToHost, streams[i]));
    }
    float t_kernel = 0, t_copy = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_kernel, t1, t2));  
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_copy, copy_start, copy_stop)); 

    std::cout << "t_kernel: " << t_kernel << " ms" << std::endl;
    std::cout << "t_copy: " << t_copy << " ms" << std::endl;

    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
    }
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }

    return 0;
}

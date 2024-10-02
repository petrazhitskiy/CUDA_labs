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
    size_t numStream; 
    if (argc > 2){
        N = atoi(argv[1]);
        numStream = atoi(argv[2]);
    }
    else if (argc > 1)
        N = atoi(argv[1]);
    else {
        N = VECTOR_SIZE;
        numStream = STREAM_NUM;
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
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, size));
    CHECK_CUDA_ERROR(cudaMemset(d_a, 0, N * sizeof(double)));

    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    int segmentSize = N / numStreams;

    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a + i * segmentSize, h_a + i * segmentSize, segmentSize * sizeof(double), cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b + i * segmentSize, h_b + i * segmentSize, segmentSize * sizeof(double), cudaMemcpyHostToDevice, streams[i]));
    }

    int threads = 128;
    int blocks = (segmentSize + threads - 1) / threads;
    for (int i = 0; i < numStreams; ++i) {
        sumvec_d<<<blocks, threads, 0, streams[i]>>>(d_a + i * segmentSize, d_b + i * segmentSize, d_c + i * segmentSize, segmentSize);
    }

    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c + i * segmentSize, d_c + i * segmentSize, segmentSize * sizeof(double), cudaMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA_ERROR(cudaFreeHost(h_a));
    CHECK_CUDA_ERROR(cudaFreeHost(h_b));
    CHECK_CUDA_ERROR(cudaFreeHost(h_c));

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }

    return 0;
}

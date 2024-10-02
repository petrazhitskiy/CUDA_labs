#include <iostream>
#include <math.h>
#include <chrono>
#include <string>
#include <cstring>

#define VECTOR_SIZE 10000000

int main(int argc, char** argv) {
    size_t N;
    if (argc > 1)
        N = atoi(argv[1]);
    else 
        N = VECTOR_SIZE;

    double *h_a, *h_b, *h_c;

    h_a = (double*)malloc(N*sizeof(double));
    h_b = (double*)malloc(N*sizeof(double));
    h_c = (double*)malloc(N*sizeof(double));
    
    if (h_a == NULL || h_b == NULL || h_c == NULL)
        std::cerr << "malloc error" << std::endl;
    memset((void*)h_c, 0, N);
    
    for (int i = 0; i < N; ++i) {
        h_a[i] = 0.1;
        h_b[i] = 0.5;
    }

    auto start_host = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) 
        h_c[i] = sin(h_a[i])*sin(h_a[i]) + cos(h_b[i])*cos(h_b[i]);
    for (int i = 0; i < 10; ++i)
        std::cout << h_c[i] << " ";
    std::cout << std::endl;
    auto end_host = std::chrono::high_resolution_clock::now();
    double t_host = std::chrono::duration<double, std::milli>(end_host - start_host).count();
    
    std::cout << "t_host : " << t_host << " ms" << std::endl;
    
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

#include <memory>
#include <iostream>

__glonal__ void sumvec_d(double* a, double* b, double *c, int N){
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (glonal_id < N)
        c[global_id] = a[global_id] + b[global_id];
    // c[0] += a[global_id] + b[global_id]; результат неопределен, гонка данных

    cudamemcpy(h_c, d_c, N * sizeof(double), cudaMemcpyDevideToHost); // синхронный вызов, хост блокируется, пока не выполнется функция  
}   

// Сложение векторов

8int main(){

    double *h_a, *h_b, *h_c;
    double *d_a, *d_b, *d_c;
    int N = 1000000;


    h_a = (double)malloc(N * sizeof(double));
    h_b = (double)malloc(N * sizeof(double));
    h_c = (double)malloc(N * sizeof(double));
    //инициализация на хосте 
    for (int i = 0 ; i < N;++i){
        h_a[i] = 1.0;
        h_b[i] = 2.0;
        h_c[i] = 0.0;
    }

    // выделяем память на устройстве &d_a (тип double **) 
    cudamalloc((void**)&d_a, N*sizeof(double)); //  выделяем память на девайсе 
    cudamalloc((void**)&d_b, N*sizeof(double)); //  выделяем память на девайсе 
    cudamalloc((void**)&d_c, N*sizeof(double)); //  выделяем память на девайсе 
    
    cudeMemcpy(d_a, h_a, N*sizeof(double), cudaMemcopyHostToDevice)
    cudeMemcpy(d_b, h_b, N*sizeof(double), cudaMemcopyHostToDevice)
    
    cudeMemset(d_c, N*sizeof(double)); // сбрасываем память чтобы не копировать 

    int threads = 128;
    int blocks = N/threads + 1
    
    sumvec_d <<<blocks, threads>>> (d_a, d_b, d_c, N); //выхзов асинхронный 
    // хост может выполнять задачи дальше 
    
    //memset(void* ptr, 0, size_t byte)  - сбросить в 0 значения массива 


    // ОБРАБОТКА ОШИБОК 

    // cudaError_r err;
    // err = cudeMallo, Memset, Memcpy - возвращают код ошибки 
    // Проверка 
    //      if (err != cudaSuccess) { char* msg = cudaGetErrorString(err);}
    //  hcernel<<...>>> (...)
    //  err = cudeGetLastError();
    //  можно написать свою функцию для проверки ошибок 
    //  void check_error(char*) {
    //      cudaError_t = ....
    //  } 
    // 
    //  можно макторосом #define CHECK_ERROR(a) err  = cudaGetLastError(); \
    //  if (err != cudeSuccess) {
    //      printf("%s(%d): %s: %s \n", \
    //          _FILE_ - имя файла, _LINE_, a, cudeGetErrorString(err)) 
    //  }           
    

    // ЗАМЕР ВРЕМЕНИ
    // 
    // cudaEvent_t t1, t2; слолько устройство чем-то занималось
    // float time;
    // cudeEventCreate(%t1); // вызовы асинхронные
    // cudeEventCreate(%t2);
    // 
    // cudeEventRecord(t1, 0);
    // ....GPU
    // cudeEventRecord(t2, 0);
    // 
    // cudeEventSynchronize(t2) или cudaDeviceSynchronize(); - глобальная синхронизация 
    // cudaEventElapsedTime(&time, t1, t2); будет записано в микросекундах
    

    // АППАРАТНАЯ ОЧЕРЕДЬ
    // GPU может производить вычисления и копирование одновременно
    // cudaStream_t st[2]
    // cudaStreamCreate(&st[0]) // инициализацтя аппаратной очереди 
    // cudaStreamCreate(&st[1])
    // 
    // НЕПРАВЛЬНО 
    // for (int i = 0; i < 2;++i)
    //      cutaMemcpyAsync(d_a + i * N / 2, h_a + i * N/2, N/2*size0f(double), st[i])
    //      cutaMemcpyAsync(d_b + i * N / 2, h_b + i * N/2, N/2*size0f(double), st[i])
    // Правильно for (i...) {hcernel<<<blocks, threads, 0, st[i]>>> (h_a+i*N/2, h_b+i*N/2,h_c+i*N/2, N/2)
    // Правильно for (i...) {cudaMemcopyAsync(h_c+i*N/2, d_a+i*N/2, N/2*sizeof(double),cudaDeviceToHost,st[i])
    // КАЖДЫЙ ЭТАП ДЕЛАЕТСЯ В ОТДЕЛЬНОМ ЦИКЛЕ 
    
    
    
    //  PINNED MEMORY 
    //  cudaMallocHost() + cudaMemcpyAsync(...)
    // 
    // 
    // 



    return 0;
}

// в куде есть атоматрные операции 
//      сложение  - atomicAdd(void* ptr, value)
//                  atomicSub();
//                  atomicExch()
//                  atomicAnd()
//                  atomicOr()






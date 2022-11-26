#include "io.h"
#include<stdlib.h>
using namespace std;

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-12;
    bool match = 1;
    for (int i=0; i<N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.12f gpu %5.12f at current %d\n",hostRef[i],gpuRef[i],i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}
// W_CHK macro is used to check if a file write is successfully or not.
#define W_CHK(call)                                         \
{                                                           \
    const int written = call;                               \
    if (written == EOF) {                                     \
        printf("error written\n");                          \
        exit(1);                                            \
    }                                                       \
}                                                           \

// CHECK macro from Grossman and McKercher, "Professional CUDA C Programming"
#define CHECK(call)                                         \
{                                                           \
    const cudaError_t error = call;                         \
    if (error != cudaSuccess) {                             \
        printf("Error: %s:%d, ", __FILE__, __LINE__);       \
        printf("code:%d, reason: %s \n",                    \
                error, cudaGetErrorString(error));          \
        exit(1);                                            \
    }                                                       \
}

__device__ double atomicAdd_my(double* address, float val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

void computeQ(float *distanceMatrix, int width, int height, float* Q){
    long start, end;
    start = getTime();
    double sum = 0.0f;
    for(int i=0; i < height; i++){
        for(int j=0; j < height; j++){
            Q[i * height + j] = 1 / (1 + distanceMatrix[i * height + j]);
            Q[i * height + i] = 0;
            sum += Q[i * height + j];
        }
    }
    for(int i=0; i < height; i++){  // 归一化
        for(int j=0; j < height; j++){
            Q[i * height + j] /= sum;
            Q[i * height + j] = fmax(1e-12, Q[i * height + j]);
        }
    }
    end = getTime();
    cout<<"the time cost by CPU is "<<end - start <<" ns\n";
}

__device__ double sum_all;

// __global__ void compute_raw_Q(float *distanceMatrix, int width, int height, float* Q)
// {
//     __shared__ float sum;
//     sum = 0.0;
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int diag = (tid / height == tid % height)? 1:0;
//     float tmp = 0.0;

//     if(diag == 0 && tid < height * height)
//     {
//         tmp = 1E6 / (1 + distanceMatrix[tid]);
//     }
//     if(tid < height * height)
//         Q[tid] = tmp;
//     atomicAdd(&sum ,tmp);
// 	__syncthreads();
        
//     if(threadIdx.x == 0)
//         atomicAdd(&sum_all ,sum);
// }   

// 优化加速明显，4倍
__global__ void compute_raw_Q(float *distanceMatrix, int width, int height, float* Q, float* sum_sequence, int cluster)
{
    extern __shared__ float tmp[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int diag = (tid / height == tid % height)? 1:0;

    if(diag == 0 && tid < height * height)
        tmp[threadIdx.x] = 1 / (1 + distanceMatrix[tid]);
    else
        tmp[threadIdx.x] = 0.0;
    if(tid < height * height)
        Q[tid] = tmp[threadIdx.x];
	__syncthreads();
    if(tid < height * height)
    {
        for (int stride=blockDim.x/2; stride>0; stride>>=1)
        { 
            if (threadIdx.x<stride)
                tmp[threadIdx.x] += tmp[threadIdx.x+stride]; 
            __syncthreads(); 
        }    
    }
    if(threadIdx.x == 0)
        atomicAdd(&sum_sequence[blockIdx.x / cluster] , tmp[0]);
        // atomicAdd_my(&sum_all , tmp[0]);
}

__global__ void sum_cluster(float * sum_sequence, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float tmp_sum[];
    if(tid < size)
        tmp_sum[threadIdx.x] = sum_sequence[tid];
    else
        tmp_sum[threadIdx.x] = 0.0;

    for (int stride=blockDim.x/2; stride>0; stride>>=1)
    { 
        if (threadIdx.x<stride)
            tmp_sum[threadIdx.x] += tmp_sum[threadIdx.x+stride]; 
        __syncthreads(); 
    }

    if(threadIdx.x == 0)
        atomicAdd_my(&sum_all , tmp_sum[0]);

}

__global__ void normalize(float* Q, int width, int heigh)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < heigh * heigh)
    {
        Q[tid] /= sum_all;
        Q[tid] = Q[tid]>1e-12? Q[tid]:1e-12; //避免分支
    }
}

void GPU_computeQ(float *distanceMatrix, int width, int height, float* Q)
{
    long start, end;

    float * d_distanceMatrix, *d_Q, *d_sum_sequence;
    CHECK(cudaMalloc((void **)&d_distanceMatrix, height * height * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Q, height * height * sizeof(float)));

    CHECK(cudaMemcpy((void *)d_distanceMatrix, (void *)distanceMatrix, height * height * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blocksize(128);
	dim3 gridsize(divup(height * height ,blocksize.x));
    int cluster = 16;
    int long_sum_sequence = divup(gridsize.x, cluster);
    CHECK(cudaMalloc((void **)&d_sum_sequence, long_sum_sequence * sizeof(float)));
    CHECK(cudaMemset((void *)d_sum_sequence, 0.0, long_sum_sequence * sizeof(float)));

    start = getTime();
    compute_raw_Q<< <gridsize, blocksize>> >(d_distanceMatrix, height, height, d_Q, d_sum_sequence, cluster);

    cudaDeviceSynchronize();

    dim3 blocksizesum(32);
    dim3 gridsizesum(divup(long_sum_sequence ,blocksizesum.x));
    sum_cluster<< <gridsizesum, blocksizesum, blocksizesum.x>> >(d_sum_sequence, long_sum_sequence);
    cudaDeviceSynchronize();

    normalize<< <gridsize, blocksize>> >(d_Q, height, height);
    cudaDeviceSynchronize();
    end = getTime();
    CHECK(cudaMemcpy((void *)Q, (void *) d_Q, height * height * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_distanceMatrix));
    CHECK(cudaFree(d_Q));


    cout<<"the time cost by GPU is "<<end - start <<" ns\n";
}

int main(int argc, char const *argv[])
{
    int height = 10000, width = 50;
    float *distanceMatrix = new float[height * height];
    float *CPU_Q = new float[height * height];
    float *GPU_Q = new float[height * height];
    for(int i=0;i < height * height;i++)
        distanceMatrix[i] = rand() % 10000;
    computeQ(distanceMatrix, width, height, CPU_Q);
    GPU_computeQ(distanceMatrix, width, height, GPU_Q);

    checkResult(CPU_Q, GPU_Q, height * height);
}

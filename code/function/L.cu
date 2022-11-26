#include "io.h"
#include<stdlib.h>
using namespace std;
__global__ void warm_up_gpu(){
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid; 
}
void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-6;
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

float computeLoss(float *P, float *Q, int height){
	long start, end;
    start = getTime();

    float loss = 0.0f;
    for(int i=0; i < height; i++){
        for(int j=0; j < height; j++){
            float p = P[i * height + j];
            float q = Q[i * height + j];
            loss += p * log(p / q);
        }
    }
	end = getTime();
    cout<<"the time cost by CPU is "<<end - start <<" ns\n";
    return loss;
}

// 开share memory继续优化
__global__ void compute_loss(float* P, float* Q, int height, float* loss)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ float buffer[];
	if(tid < height * height)
	{
		float p = P[tid];
		float q = Q[tid];
		buffer[threadIdx.x] = p * log(p / q);
	    // __syncthreads();

        // for (int stride=blockDim.x/2; stride>0; stride>>=1)
        // { 
        //     if (threadIdx.x<stride)
        //     {
        //         buffer[threadIdx.x] += buffer[threadIdx.x+stride]; 
        //     }
        //     __syncthreads(); 
        // }

        // if (threadIdx.x == 0)
        // {
            atomicAdd(loss, buffer[threadIdx.x]);
        // }
    }
}


void Cuda_computeloss(float *P, float *Q, int height, float *result)
{
	long start, end;

    float * d_P, * d_Q, * d_loss;
    CHECK(cudaMalloc((void **)&d_P, height * height * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_Q, height * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_loss, sizeof(float)));

    CHECK(cudaMemcpy((void *)d_P, (void *)P, height * height * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy((void *)d_Q, (void *)Q, height * height * sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMemset((void *)d_loss, 0.0, sizeof(float))); 
    dim3 blocksize(256);
	dim3 gridsize(divup(height * height ,blocksize.x));
	warm_up_gpu<< <gridsize, blocksize>> >();
	start = getTime();
	compute_loss<< <gridsize, blocksize, blocksize.x>> >(d_P, d_Q, height, d_loss);
	end = getTime();
    cudaDeviceSynchronize();

    CHECK(cudaMemcpy((void *)result, (void *)d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_P));
    CHECK(cudaFree(d_Q));

    cout<<"the time cost by GPU is "<<end - start <<" ns\n";
}

int main(int argc, char const *argv[])
{
    int height = 10000, width = 50;
    float *P = new float[height * height];
    float *Q = new float[height * height];
	float loss_cpu, loss_gpu;
    for(int i=0;i < height * height;i++)
	{
        P[i] = (rand() % 10) + 1.0 / 1000000.0;
		Q[i] = (rand() % 10) + 1.0 / 1000000.0;
	}
    loss_cpu = computeLoss(P, Q, height);

	Cuda_computeloss(P, Q, height, &loss_gpu);

	printf("result of CPU %lf, result of GPU %lf\n",loss_cpu, loss_gpu);
}

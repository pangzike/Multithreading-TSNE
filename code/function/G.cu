#include "io.h"
#include<stdlib.h>
using namespace std;

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 10;
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

void computeGradient(float *P, float *Q, float *Y, float *distanceMatrix, int height, float *gradient, int reducedDim){
    long start, end;
    start = getTime();
    static int step = 0;
    step += 1;
    float scale = step > 50? 1.0f : 4.0f;
    // float scale = 4.0f;
    for(int i=0; i < height; i++){
        for(int k=0; k < reducedDim; k++){
            gradient[i * reducedDim + k] = 0.0f;
        }
        for(int j=0; j < height; j++){
            float Pij = P[i * height + j] * scale;  // 乘以scale可以加速收敛
            float Qij = Q[i * height + j];
            float temp = 1 + distanceMatrix[i * height + j];
            for(int k=0; k < reducedDim; k++){
                float Yik = Y[i * reducedDim + k];
                float Yjk = Y[j * reducedDim + k];
                gradient[i * reducedDim + k] += 4 * (Pij - Qij) * (Yik - Yjk) / temp;
            }
        }
    }
    end = getTime();
    cout<<"the time cost by CPU is "<<end - start <<" ns\n";
}

void computeGradient_with_half(float *P, float *Q, float *Y, float *distanceMatrix, int height, float *gradient, int reducedDim)
{
    
}
// 动态并行
// __global__ void computegradient(float *P, float *Q, float *Y, float *distanceMatrix, int height, float *gradient, int reducedDim, int scale, int span)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int i = tid / height;
//     int j = tid % height;
//     if(i < height && j < height)
//     {
//         float Pij = P[i * height + j] * scale;
//         float Qij = Q[i * height + j];
//         float temp = 1 + distanceMatrix[i * height + j];
//         for(int k=0; k < reducedDim; k++){
//             float Yik = Y[i * reducedDim + k];
//             float Yjk = Y[j * reducedDim + k];
//             float cur_gradient = 4 * (Pij - Qij) * (Yik - Yjk) / temp;
//             atomicAdd(&gradient[i * reducedDim + k] , cur_gradient);
//         }
//     }
// }

// 提速明显，70倍
__global__ void computegradient(float *P, float *Q, float *Y, float *distanceMatrix, int height, float *gradient, int reducedDim, int scale, int span)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = tid / span;
    int j = tid % span;
    extern __shared__ float gradient_buffer[][2];
    if(j < height && i < height)
    {
        float Pij = P[i * height + j] * scale;
        float Qij = Q[i * height + j];
        float temp = 1 + distanceMatrix[i * height + j];
        float Yik = Y[i * reducedDim];
        float Yjk = Y[j * reducedDim];
        gradient_buffer[threadIdx.x][0] = 4 * (Pij - Qij) * (Yik - Yjk) / temp;
        Yik = Y[i * reducedDim + 1];
        Yjk = Y[j * reducedDim + 1];
        gradient_buffer[threadIdx.x][1] = 4 * (Pij - Qij) * (Yik - Yjk) / temp;
    }
    else
    {
        gradient_buffer[threadIdx.x][0] = 0;
        gradient_buffer[threadIdx.x][1] = 0;
    }
    __syncthreads();
    for (int stride=blockDim.x/2; stride>0; stride>>=1)
    { 
        if (threadIdx.x<stride)
        {
            gradient_buffer[threadIdx.x][0] += gradient_buffer[threadIdx.x+stride][0];
            gradient_buffer[threadIdx.x][1] += gradient_buffer[threadIdx.x+stride][1]; 
        }
        __syncthreads(); 
    }

    if(threadIdx.x == 0)
    {
        atomicAdd(&gradient[i * reducedDim], gradient_buffer[0][0]);
        atomicAdd(&gradient[i * reducedDim + 1], gradient_buffer[0][1]);
    }
}

void Cuda_computeGradient(float *P, float *Q, float *Y, float *distanceMatrix, int height, float *gradient, int reducedDim)
{
	long start, end;
    
    static int step = 0;
    step += 1;
    float scale = step > 50? 1.0f : 4.0f;
    // float scale = 4.0f;
    float * d_P, * d_Q, * d_Y, *d_distanceMatrix, * d_gradient;
    CHECK(cudaMalloc((void **)&d_P, height * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_Q, height * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_Y, 2 * height * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_distanceMatrix, height * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_gradient, 2 * height * sizeof(float)));

    CHECK(cudaMemcpy((void *)d_P, (void *) P, height * height * sizeof(float), cudaMemcpyHostToDevice));
   	CHECK(cudaMemcpy((void *)d_Q, (void *) Q, height * height * sizeof(float), cudaMemcpyHostToDevice));
   	CHECK(cudaMemcpy((void *)d_Y, (void *) Y, 2 * height * sizeof(float), cudaMemcpyHostToDevice));
   	CHECK(cudaMemcpy((void *)d_distanceMatrix, (void *) distanceMatrix, height * height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset((void *)d_gradient, 0, sizeof(float) * 2 * height)); 
    dim3 blocksize(64);
	// dim3 gridsize(divup(height * height ,blocksize.x));
    dim3 gridsize(divup(height * divup(height, blocksize.x) * blocksize.x, blocksize.x));
    start = getTime();
    computegradient<< <gridsize, blocksize, blocksize.x>> >(d_P, d_Q, d_Y, d_distanceMatrix, height, d_gradient, reducedDim, scale, divup(height, blocksize.x) * blocksize.x);
    cudaDeviceSynchronize();
    end = getTime();
    CHECK(cudaMemcpy((void *)gradient, (void *) d_gradient, 2 * height * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_distanceMatrix));
    CHECK(cudaFree(d_P));
	CHECK(cudaFree(d_Q));
	CHECK(cudaFree(d_Y));
	CHECK(cudaFree(d_gradient));

    
    cout<<"the time cost by GPU is "<<end - start <<" ns\n";
}

int main(int argc, char const *argv[])
{
    int height = 10000, width = 50;
    float *P = new float[height * height];
    float *Q = new float[height * height];
	float *reducedMatrix = new float[2 * height];
	float *distanceMatrix = new float[height * height];
    float *GPU_G = new float[2 * height];
	float *CPU_G = new float[2 * height];
    for(int i=0;i < height * height;i++)
	{
        distanceMatrix[i] = rand() % 10;
		P[i] = rand() % 10;
		Q[i] = rand() % 10;
	}
	for(int i=0;i < 2 * height;i++)
	{
		reducedMatrix[i] = rand() % 10;
	}
    
	computeGradient(P, Q, reducedMatrix, distanceMatrix, height, CPU_G, 2);

    Cuda_computeGradient(P, Q, reducedMatrix, distanceMatrix, height, GPU_G, 2);
	

    checkResult(CPU_G, GPU_G, 2 * height);
}

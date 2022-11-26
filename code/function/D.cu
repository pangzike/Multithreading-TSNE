#include "io.h"
#include<stdlib.h>
#include<random>
using namespace std;

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

void computeDistanceMatrix(float* matrix, int width, int height, float* distanceMatrix) {
    
    long start, end;
    start = getTime();
	
	for(int i=0; i < height; i++){
        for(int j=0; j < height; j++){
            // 求两个向量之间的距离
            float distance = 0.0f;
            for(int k=0; k < width; k++){
                float temp = matrix[i * width + k] - matrix[j * width + k];  // 作差
                distance += temp * temp;
            }
            distanceMatrix[i * height + j] = distance;  // 求平方和
        }
    }
    end = getTime();
    cout<<"the time cost by GPU is "<<end - start <<" ns\n";
}

// 开share memory继续优化
__global__ void compute_distance(float* matrix, int width, int height, float* distanceMatrix)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int i = tid / height;
	int j = tid % height;
	float distance = 0.0;
	for(int k=0;k <width;k++)
	{
		float temp = matrix[i * width + k] - matrix[j * width + k];
		distance += temp * temp;
	}
	distanceMatrix[i * height + j] = distance;
}

// 速度比上面慢了4倍，可能是线程开的太多了，开销太大
// __global__ void compute_distance(float* matrix, int width, int height, float* distanceMatrix)
// {
// 	long tid = threadIdx.x + blockDim.x * blockIdx.x;
// 	int i = blockIdx.x / height;
// 	int j = blockIdx.x % height;
//     extern __shared__ float distance[];

//     float tmp;
//     tmp = matrix[i * width + threadIdx.x] - matrix[j * width + threadIdx.x];
//     distance[threadIdx.x] = tmp * tmp;
//     __syncthreads();
//     for (int stride=blockDim.x/2; stride>0; stride>>=1)
//     { 
//     	if (threadIdx.x<stride)
//     		distance[threadIdx.x] += distance[threadIdx.x+stride]; 
//     		__syncthreads(); 
//     }
//     if(threadIdx.x == 0)
//         distanceMatrix[blockIdx.x] = distance[0];
// }

void Cuda_computeDistanceMatrix(float* matrix, long width, long height, float* distanceMatrix)
{
	long start, end;
    

    float * d_matrix, * d_distanceMatrix;
    CHECK(cudaMalloc((void **)&d_matrix, height * width * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_distanceMatrix, height * height * sizeof(float)));

    CHECK(cudaMemcpy((void *)d_matrix, (void *)matrix, width * height * sizeof(float), cudaMemcpyHostToDevice));
    dim3 blocksize(64);
    // dim3 blocksize(32);
	dim3 gridsize(divup(height * height,blocksize.x));
    start = getTime();
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	compute_distance<< <gridsize, blocksize, blocksize.x>> >(d_matrix, width, height, d_distanceMatrix);
    cudaDeviceSynchronize();
    end = getTime();
    CHECK(cudaMemcpy((void *)distanceMatrix, (void *) d_distanceMatrix, height * height * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_distanceMatrix));
    CHECK(cudaFree(d_matrix));

    
    cout<<"the time cost by GPU is "<<end - start <<" ns\n";
}

int main(int argc, char const *argv[])
{
    int height = 1000, width = 2;
    float *matrix = new float[height * width];
    float *CPU_D = new float[height * height];
    float *GPU_D = new float[height * height];
    unsigned seed = 0;
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for(int i=0;i < height * width;i++)
        matrix[i] = distribution(generator);
    
	computeDistanceMatrix(matrix, width, height, CPU_D);

	Cuda_computeDistanceMatrix(matrix, width, height, GPU_D);

    checkResult(CPU_D, GPU_D, height * height);
}

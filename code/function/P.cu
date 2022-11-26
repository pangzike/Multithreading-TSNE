#include "io.h"
#include<stdlib.h>
#include<omp.h>
#include<random>
using namespace std;

/**
 * 计算给定向量的熵
 * @distance: 距离向量
 * @width: 向量的长度
 * @sigma: 方差
 * @k: 表示正在处理第k个向量，在计算熵的时候，是不考虑向量第k个元素的
 * @e: 中间变量
 * @s: 中间变量
 */
float computeEntropy(float* vector, int dim, float sigma, int k, float* e, float* s){
    float entropy = 0.0f;
    float temp = 0.0f;

    *s = 0.0f;
    for(int i=0; i < dim; i++){
        float distance = vector[i];
        e[i] = exp(- distance / sigma);
        e[k] = 0.0f;
        *s += e[i];
        temp += e[i] * distance;
    }
    entropy = log(*s) + temp / (*s * sigma);
    return entropy;
}

/**
 * 计算一个矩阵的行向量两两之间的距离的平方
 * @matrix: 矩阵数组
 * @width: 矩阵宽度
 * @height: 矩阵高度
 * @distanceMatirx: 保存输出的距离矩阵
 */
void computeDistanceMatrix(float* matrix, int width, int height, float* distanceMatrix) {
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
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-7;
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

/**
 * 计算P，核心部分是使用二分查找找到sigma
 * @matrix: 矩阵数组
 * @width: 矩阵宽度
 * @height: 矩阵高度
 * @perplexity: 指定的困惑度
 * @P: 保存最后求出来的P
 */
void computeP(float *matrix, int width, int height, float perplexity, float* P){
	long start, end;
    start = getTime();
    float targetEntropy = log(perplexity);
    if(targetEntropy > log(height - 1)){
        std::cout << "The target entropy is " << targetEntropy
                  << " while the maximum entropy is " << log(height - 1) << std::endl;
        exit(0);
    }

    float *distanceMatrix = new float[height * height];
    computeDistanceMatrix(matrix, width, height, distanceMatrix);

    float MIN = 0.0f, MAX = std::numeric_limits<float>::max();
    float *e = new float[height], s, currentEntropy;

    for(int i=0; i < height; i++){
        // 初始化
        float max = MAX, min = MIN;  // 二分法的上下界
        float sigma = 0.0f;  // 这个初始化很重要，如果sigma初始比较小，可能导致后面算指数时全部下溢为0，然后0作为分子得到nan
        for(int j=0; j < height; j++){  // 使用平均距离来初始化
            sigma += distanceMatrix[i * height + j];
        }
        sigma /= height;
        currentEntropy = computeEntropy(distanceMatrix + i * height, height, sigma, i, e, &s);
        // 二分法查找
        int step = 0;
        while(abs(currentEntropy - targetEntropy) > 1e-5 and step < 50){
            if(currentEntropy > targetEntropy){  // 困惑度太大，说明sigma太大
                max = sigma;
                sigma = (min == MIN)? sigma / 2: (min + max) / 2;
            } else {  // 困惑度太小，说明sigma太小
                min = sigma;
                sigma = (max == MAX)? sigma * 2: (min + max) / 2;
            }
            currentEntropy = computeEntropy(distanceMatrix + i * height, height, sigma, i, e, &s);
            step += 1;
        }
        // 查找完毕，求出P
        for(int j=0; j < height; j++){
            P[i * height + j] = e[j] / s;
        }
    }
    // 对称化
    for(int i=0; i < height; i++){
        for(int j=i; j < height; j++){
            float Pij = P[i * height + j];
            float Pji = P[j * height + i];
            P[i * height + j] = fmax(1e-12, (Pij + Pji) / (2 * height));
            P[j * height + i] = fmax(1e-12, (Pij + Pji) / (2 * height));
        }
    }
    delete[] distanceMatrix;
    delete[] e;
	end = getTime();
    cout<<"the time cost by CPU is "<<end - start <<" ns\n";
}

__global__ void kernal_compute_distance(float* matrix, int width, int height, float* distanceMatrix)
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

void computeP_openmp(float *matrix, int width, int height, float perplexity, float* P)
{
	long start, end;
    start = getTime();
    float targetEntropy = log(perplexity);
    if(targetEntropy > log(height - 1)){
        std::cout << "The target entropy is " << targetEntropy
                  << " while the maximum entropy is " << log(height - 1) << std::endl;
        exit(0);
    }

    float *distanceMatrix = new float[height * height];
	float * d_matrix, * d_P, *d_distanceMatrix;
    CHECK(cudaMalloc((void **)&d_matrix, width * height* sizeof(float)));
    CHECK(cudaMalloc((void **)&d_P, height * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_distanceMatrix, height * height * sizeof(float)));

    CHECK(cudaMemcpy((void *)d_matrix, (void *)matrix, width * height * sizeof(float), cudaMemcpyHostToDevice));
	
	dim3 blocksize(64);
	dim3 gridsize(divup(height * height,blocksize.x));
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	kernal_compute_distance<< <gridsize, blocksize, blocksize.x>> >(d_matrix, width, height, d_distanceMatrix);
    cudaDeviceSynchronize();

	CHECK(cudaMemcpy((void *)distanceMatrix, (void *)d_distanceMatrix, height * height * sizeof(float), cudaMemcpyDeviceToHost));

    float MIN = 0.0f, MAX = std::numeric_limits<float>::max();

#pragma omp parallel for num_threads(32)
    for(int i=0; i < height; i++){
        // 初始化
		float *e = new float[height], s, currentEntropy;
        float max = MAX, min = MIN;  // 二分法的上下界
        float sigma = 0.0f;  // 这个初始化很重要，如果sigma初始比较小，可能导致后面算指数时全部下溢为0，然后0作为分子得到nan
        for(int j=0; j < height; j++){  // 使用平均距离来初始化
            sigma += distanceMatrix[i * height + j];
        }
        sigma /= height;
        currentEntropy = computeEntropy(distanceMatrix + i * height, height, sigma, i, e, &s);
        // 二分法查找
        int step = 0;
        while(abs(currentEntropy - targetEntropy) > 1e-5 and step < 50){
            if(currentEntropy > targetEntropy){  // 困惑度太大，说明sigma太大
                max = sigma;
                sigma = (min == MIN)? sigma / 2: (min + max) / 2;
            } else {  // 困惑度太小，说明sigma太小
                min = sigma;
                sigma = (max == MAX)? sigma * 2: (min + max) / 2;
            }
            currentEntropy = computeEntropy(distanceMatrix + i * height, height, sigma, i, e, &s);
            step += 1;
        }
        // 查找完毕，求出P
        for(int j=0; j < height; j++){
            P[i * height + j] = e[j] / s;
        }
		delete[] e;
    }
    // 对称化
    for(int i=0; i < height; i++){
        for(int j=i; j < height; j++){
            float Pij = P[i * height + j];
            float Pji = P[j * height + i];
            P[i * height + j] = fmax(1e-12, (Pij + Pji) / (2 * height));
            P[j * height + i] = fmax(1e-12, (Pij + Pji) / (2 * height));
        }
    }
    delete[] distanceMatrix;
	end = getTime();
    cout<<"the time cost by CPU is "<<end - start <<" ns\n";
}


__global__ void kernal_compute_sigma(float * distance, int height , float* sigma, int swift)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ float tmp_distance[];
	if( tid < height)
		tmp_distance[threadIdx.x] = distance[swift * height + tid];
	__syncthreads();
	for (int stride=blockDim.x/2; stride>0; stride>>=1)
	{ 
		if (threadIdx.x<stride)
			tmp_distance[threadIdx.x] += tmp_distance[threadIdx.x+stride]; 
		__syncthreads(); 
	}
	if (threadIdx.x==0) 
		atomicAdd(sigma, tmp_distance[0] / height);
}

__global__ void kernal_compute_entropy(float* vector, int dim, float* sigma, int k, float* e, float * out)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ float tmp_e[][2];
	if(tid < dim && tid != k)
	{
		float distance = vector[k * dim + tid];
		tmp_e[threadIdx.x][0] = exp(- distance / *sigma);
		tmp_e[threadIdx.x][1] = tmp_e[threadIdx.x][0] * distance;
		e[tid] = tmp_e[threadIdx.x][0];
	}
	else
	{
		tmp_e[threadIdx.x][0] = 0;
		tmp_e[threadIdx.x][1] = 0;
	}
	__syncthreads();
	for (int stride=blockDim.x/2; stride>0; stride>>=1)
    { 
        if (threadIdx.x<stride)
        {
            tmp_e[threadIdx.x][0] += tmp_e[threadIdx.x+stride][0];
            tmp_e[threadIdx.x][1] += tmp_e[threadIdx.x+stride][1]; 
        }
        __syncthreads(); 
    }
	if(threadIdx.x == 0)
    {
        atomicAdd(&out[0], tmp_e[0][0]);
        atomicAdd(&out[1], tmp_e[0][1]);
    }
}

__global__ void kernal_compute_p(float * P, float * e, float * s, int dim, int k)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	P[k * dim + tid] = e[tid] / s[0];
}

void Cuda_computeP(float *matrix, int width, int height, float perplexity, float* P)
{
	float targetEntropy = log(perplexity);
    if(targetEntropy > log(height - 1)){
        std::cout << "The target entropy is " << targetEntropy
                  << " while the maximum entropy is " << log(height - 1) << std::endl;
        exit(0);
    }

	long start, end;
    start = getTime();

    float * d_matrix, * d_P, *d_distanceMatrix;
    CHECK(cudaMalloc((void **)&d_matrix, width * height* sizeof(float)));
    CHECK(cudaMalloc((void **)&d_P, height * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_distanceMatrix, height * height * sizeof(float)));

    CHECK(cudaMemcpy((void *)d_matrix, (void *)matrix, width * height * sizeof(float), cudaMemcpyHostToDevice));
	
	dim3 blocksize(64);
	dim3 gridsize(divup(height * height,blocksize.x));
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	kernal_compute_distance<< <gridsize, blocksize, blocksize.x>> >(d_matrix, width, height, d_distanceMatrix);
    cudaDeviceSynchronize();

	float MIN = 0.0f, MAX = std::numeric_limits<float>::max();


	for(int i=0; i < height; i++){
        // 初始化
        float max = MAX, min = MIN;  // 二分法的上下界
        float sigma = 0.0f;  // 这个初始化很重要，如果sigma初始比较小，可能导致后面算指数时全部下溢为0，然后0作为分子得到nan

		float * d_sigma;
		CHECK(cudaMalloc((void **)&d_sigma, sizeof(float)));
		CHECK(cudaMemset((void *)d_sigma, 0, sizeof(float)));
		dim3 blocksize_sum(128);
		dim3 gridsize_sum(divup(height, blocksize.x));
		kernal_compute_sigma<< <gridsize_sum, blocksize_sum, blocksize_sum.x>> >(d_distanceMatrix, height, d_sigma, i);
		cudaDeviceSynchronize();
		CHECK(cudaMemcpy((void *)&sigma, (void *)d_sigma, sizeof(float), cudaMemcpyDeviceToHost));

		float *d_out, currentEntropy, *d_e, *out = new float[2];
		CHECK(cudaMalloc((void **)&d_out, 2 * sizeof(float)));
		CHECK(cudaMalloc((void **)&d_e, sizeof(float) * height));
		
       	dim3 blocksize_entropy(128);
		dim3 gridsize_entropy(divup(height, blocksize_entropy.x));
		CHECK(cudaMemset((void *)d_out, 0.0, 2 * sizeof(float)));
		kernal_compute_entropy<< <gridsize_entropy, blocksize_entropy, blocksize_entropy.x>> >(d_distanceMatrix, height, d_sigma, i, d_e, d_out);
		cudaDeviceSynchronize();
		CHECK(cudaMemcpy((void *)out, (void *)d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost));
		currentEntropy = log(out[0]) + out[1] / (out[0] * sigma);
        // 二分法查找
        int step = 0;
        while(abs(currentEntropy - targetEntropy) > 1e-5 and step < 50){
            if(currentEntropy > targetEntropy){  // 困惑度太大，说明sigma太大
                max = sigma;
                sigma = (min == MIN)? sigma / 2: (min + max) / 2;
            } else {  // 困惑度太小，说明sigma太小
                min = sigma;
                sigma = (max == MAX)? sigma * 2: (min + max) / 2;
            }
            CHECK(cudaMemset((void *)d_out, 0.0, 2 * sizeof(float)));
			kernal_compute_entropy<< <gridsize, blocksize>> >(d_distanceMatrix, height, d_sigma, i, d_e, d_out);
			cudaDeviceSynchronize();
			CHECK(cudaMemcpy((void *)out, (void *)d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost));
			currentEntropy = log(out[0]) + out[1] / (out[0] * sigma);
            step += 1;
        }
        // 查找完毕，求出P
        dim3 blocksize_p(1024);
		dim3 gridsize_p(divup(height, blocksize_p.x));
		kernal_compute_p<< <gridsize_p, blocksize_p>> >(d_P, d_e, d_out, height, i);
		cudaDeviceSynchronize();
    }


    CHECK(cudaMemcpy((void *)P, (void *)d_P, height * height * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_matrix));
	CHECK(cudaFree(d_P));

    end = getTime();
    cout<<"the time cost by GPU is "<<end - start <<" ns\n";
}

int main(int argc, char const *argv[])
{
    int height = 10000, width = 50, perplexity = 25;

	float * matrix = read("../train60000dim50.txt", width, height);
	float * cpu_P = new float[height * height];
	float * gpu_P = new float[height * height];

	computeP(matrix, width, height, perplexity, cpu_P);

	computeP_openmp(matrix, width, height, perplexity, gpu_P);
    checkResult(cpu_P, gpu_P, height * height);
}

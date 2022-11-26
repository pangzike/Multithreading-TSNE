#include "io.h"
#include<stdlib.h>
using namespace std;

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

void update(float *x, float *m, float *g, float *gradient, int reducedDim, int height, float lr, float momentum, float weightDecay){
    long start, end;
    start = getTime();

	for(int i=0; i < reducedDim * height; i++){
        float grad = gradient[i];
        m[i] = (momentum) * m[i] + (1 - momentum) * grad;
        g[i] = (momentum) * g[i] + momentum * grad * grad;
        x[i] = (1 - weightDecay) * x[i] - lr / (sqrt(g[i]) + 1e-12) * m[i];
    }
    end = getTime();
    cout<<"the time cost by CPU is "<<end - start <<" ns\n";
}


__global__ void kernal_updata(float *x, float *m, float *g, float *gradient, int reducedDim, int height, float lr, float momentum, float weightDecay)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < reducedDim * height)
	{
		float grad = gradient[tid];
		float tmp_g, tmp_m;
		tmp_m = (momentum) * m[tid] + (1 - momentum) * grad;
		tmp_g = (momentum) * g[tid] + momentum * grad * grad;
        g[tid] = tmp_g;
		m[tid] = tmp_m;
        x[tid] = (1 - weightDecay) * x[tid] - lr / (sqrt(tmp_g) + 1e-5) * tmp_m;
	}

}

void Cuda_update(float *x, float *m, float *g, float *gradient, int reducedDim, int height, float lr, float momentum, float weightDecay)
{
	long start, end;
    start = getTime();

    float * d_x, * d_m, *d_g, *d_gradient;
    CHECK(cudaMalloc((void **)&d_x, 2 * height* sizeof(float)));
    CHECK(cudaMalloc((void **)&d_m, 2 * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_g, 2 * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_gradient, 2 * height * sizeof(float)));

    CHECK(cudaMemcpy((void *)d_x, (void *)x, 2 * height * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy((void *)d_m, (void *)m, 2 * height * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy((void *)d_g, (void *)g, 2 * height * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy((void *)d_gradient, (void *)gradient, 2 * height * sizeof(float), cudaMemcpyHostToDevice));
    dim3 blocksize(512);
	dim3 gridsize(divup(2 * height ,blocksize.x));

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	kernal_updata<< <gridsize, blocksize>> >(d_x, d_m, d_g, d_gradient, reducedDim, height, lr, momentum, weightDecay);
    cudaDeviceSynchronize();

    CHECK(cudaMemcpy((void *)x, (void *)d_x, 2 * height * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_m));
	CHECK(cudaFree(d_g));
	CHECK(cudaFree(d_gradient));

    end = getTime();
    cout<<"the time cost by GPU is "<<end - start <<" ns\n";
}

int main(int argc, char const *argv[])
{
    int height = 1000, width = 50;
	float lr = 1, momentum = 0.9, weightDecay = 0.001;
    float * x_cpu = new float[height * 2];
	float * x_gpu = new float[height * 2];
    float * m_cpu = new float[height * 2];
	float * m_gpu = new float[height * 2];
    float * g_cpu = new float[height * 2];
	float * g_gpu = new float[height * 2];
	float * gradient = new float[height * 2];
    for(int i=0;i < height * 2;i++)
	{
        x_cpu[i] = rand() % 10;
		m_cpu[i] = rand() % 10;
		g_cpu[i] = rand() % 10;
		gradient[i] = rand() % 10;
		x_gpu[i] = x_cpu[i];
		m_gpu[i] = m_cpu[i];
		g_gpu[i] = g_cpu[i];
	}
	update(x_cpu, m_cpu, g_cpu, gradient, 2, height, lr, momentum, weightDecay);
	Cuda_update(x_gpu, m_gpu, g_gpu, gradient, 2, height, lr, momentum, weightDecay);

    checkResult(x_cpu, x_gpu, 2 * height);
}

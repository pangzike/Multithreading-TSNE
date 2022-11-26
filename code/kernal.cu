//**************************************************************************
// 文件说明：本文件为本次作业的计算TSNE的所有核函数
//**************************************************************************

/**
 * 计算一个矩阵的行向量两两之间的距离的平方
 * @matrix: 矩阵数组
 * @width: 矩阵宽度
 * @height: 矩阵高度
 * @distanceMatirx: 保存输出的距离矩阵
 */
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

/**
 * 重载实现 atomicAdd 的float加double版
 */
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

__device__ double sum_all;

/**
 * 计算Q
 * @distanceMatrix: 距离矩阵
 * @width: 矩阵宽度
 * @height: 矩阵高度
 * @Q: 保存最后求出来的Q
 * @sum_sequence: 记录每个cluster的结果
 * @cluster: cluster 的大小
 */
__global__ void kernal_compute_raw_Q(float *distanceMatrix, int width, int height, float* Q, float* sum_sequence, int cluster)
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

/**
 * 计算Q
 * @sum_sequence: 记录每个cluster的结果
 * @size: 总共有多少个cluster
 */
__global__ void kernal_sum_cluster(float * sum_sequence, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ double tmp_sum[];
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

/**
 * 计算Q
 * @width: 矩阵宽度
 * @height: 矩阵高度
 * @Q: 保存最后求出来的Q
 */
__global__ void kernal_normalize(float* Q, int width, int heigh)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < heigh * heigh)
    {
        Q[tid] /= sum_all;
        Q[tid] = Q[tid]>1e-9? Q[tid]:1e-9; //避免分支
    }
}

/**
 * 计算梯度
 * @P: P矩阵
 * @Q: Q矩阵
 * @Y: 降维后的数据，即自变量
 * @distanceMatrix: Y中向量两两之间的距离矩阵
 * @width: 降维前的宽度
 * @height: 矩阵高度
 * @gradient: 梯度
 * @reducedDim: 降维后的维度
 * @scale: P的系数
 * @span: 一行线程的数量
 */
__global__ void kernal_computegradient(float *P, float *Q, float *Y, float *distanceMatrix, int height, float *gradient, int reducedDim, int scale, int span)
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

/**
 * 使用Adam算法进行更新
 */
__global__ void kernal_update(float *x, float *m, float *g, float *gradient, int reducedDim, int height, float lr, float momentum, float weightDecay)
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
        // x[tid] = x[tid] - lr * grad;
	}

}

/**
 * 计算loss，即P和Q之间的KL散度
 */
__global__ void kernal_compute_loss(float* P, float *Q, int height, float* loss)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ float buffer[];
	if(tid < height * height)
	{
		float p = P[tid];
		float q = Q[tid];
		buffer[threadIdx.x] = p * log(p / q);
		__syncthreads();
		for (int stride=blockDim.x/2; stride>0; stride>>=1)
		{ 
			if (threadIdx.x<stride)
				buffer[threadIdx.x] += buffer[threadIdx.x+stride]; 
			__syncthreads(); 
		}
	}
	if (threadIdx.x==0) 
		atomicAdd(loss, buffer[0]);
}
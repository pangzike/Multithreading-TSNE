//**************************************************************************
// 文件说明：本文件为本次作业的计算TSNE实现函数
//**************************************************************************
#include<iostream>
#include<limits>
#include<random>
#include<math.h>
#include "kernal.cu"
#include "io.h"
using namespace std;

void tSNE(float* matrix, int width, int height, float* reducedMatrix, int reducedDim);
void computeP(float *matrix, int width, int height, float perplexity, float* P);
float computeEntropy(float* vector, int dim, float sigma, int k, float* e, float* s);

/**
 * @matrix: 矩阵数组，每行是一个要降维的向量
 * @width: 矩阵宽度（向量维度）
 * @height: 矩阵高度
 * @reducedMatrix: 保存降维后的矩阵
 * @reducedDim: 降维后的维度
 */
void tSNE(float *matrix, int width, int height, float *reducedMatrix, int reducedDim){
    // 超参数
    float perplexity=25, lr=1, stepNum=5, momentum=0.8, weightDecay=0.001f;
    // 困惑度越大，降维后的点越分散
    // 计算P
    float *P = new float[height * height];
    long timestamp1, timestamp2, timestamp3, timestamp4, timestamp5;

    timestamp1 = getTime();
    computeP(matrix, width, height, perplexity, P);
    timestamp2 = getTime();

    cout<<"compute P cost: "<< timestamp2 - timestamp1 << " ns\n";
    // 随机初始化
    unsigned seed = 0;
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for(int i=0; i < reducedDim * height; i++){
        reducedMatrix[i] = distribution(generator);
    }
    // 梯度下降
    float *Q = new float[height * height];
    float *gpu_Q = new float[height * height];
    float *gradient = new float[reducedDim * height];
    float *distanceMatrix = new float[height * height];
    float *m = new float[reducedDim * height];  // 梯度的指数加权平均
    float *g = new float[reducedDim * height];  // 梯度平方的指数加权平均，用于Adam算法
    for(int i = 0; i < reducedDim * height; i++){
        m[i] = 0;
        g[i] = 0;
    }

	float *d_P, *d_Q, *d_reducedMatrix, *d_distanceMatrix, *d_m, *d_g, *d_gradient, *d_sum_sequence;

	//分配内存
    CHECK(cudaMalloc((void **)&d_distanceMatrix, height * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_reducedMatrix, height * reducedDim * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_P, height * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_Q, height * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_gradient, reducedDim * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_m, reducedDim * height * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_g, reducedDim * height * sizeof(float)));
    int cluster = 4;
    int long_sum_sequence = divup(divup(height * height ,128), cluster);
    CHECK(cudaMalloc((void **)&d_sum_sequence, long_sum_sequence * sizeof(float)));

	//从host复制
	CHECK(cudaMemcpy((void *)d_reducedMatrix, (void *)reducedMatrix, reducedDim * height * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy((void *)d_P, (void *)P, height * height * sizeof(float), cudaMemcpyHostToDevice));

	//初始化为0
	CHECK(cudaMemset((void *)d_gradient, 0.0, sizeof(float) * reducedDim * height)); 
	CHECK(cudaMemset((void *)d_m, 0.0, sizeof(float) * reducedDim * height)); 
	CHECK(cudaMemset((void *)d_g, 0.0, sizeof(float) * reducedDim * height)); 
	float scale, *d_loss;

	CHECK(cudaMalloc((void **)&d_loss, sizeof(float)));

    for(int step=0; step < stepNum; step++){
        cout<< "step "<<step<<" : "<<endl;
// 计算低维距离
        timestamp1 = getTime();
        dim3 blocksize_d(64);
		dim3 gridsize_d(divup(height * height, blocksize_d.x));
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		kernal_compute_distance<< <gridsize_d, blocksize_d, blocksize_d.x>> >(d_reducedMatrix, reducedDim, height, d_distanceMatrix);
		cudaDeviceSynchronize();
        timestamp2 = getTime();
        cout<<"compute low dimension distance cost: "<< timestamp2 - timestamp1 << " ns\n";

        double tmp = 0.0;
        dim3 blocksize(128);
        dim3 gridsize(divup(height * height ,blocksize.x));
        CHECK(cudaMemset((void *)d_sum_sequence, 0.0, long_sum_sequence * sizeof(float)));

        cudaMemcpyToSymbol(sum_all, &tmp, sizeof(double));
        cudaDeviceSynchronize();
        kernal_compute_raw_Q<< <gridsize, blocksize>> >(d_distanceMatrix, height, height, d_Q, d_sum_sequence, cluster);
        cudaDeviceSynchronize();
        dim3 blocksizesum(32);
        dim3 gridsizesum(divup(long_sum_sequence ,blocksizesum.x));
        kernal_sum_cluster<< <gridsizesum, blocksizesum, blocksizesum.x>> >(d_sum_sequence, long_sum_sequence);
        cudaDeviceSynchronize();
        kernal_normalize<< <gridsize, blocksize>> >(d_Q, height, height);
        cudaDeviceSynchronize();

        timestamp3 = getTime();
        cout<<"compute Q cost: "<< timestamp3 - timestamp2 << " ns\n";

		scale = step > 50? 1.0f : 4.0f;
		CHECK(cudaMemset((void *)d_gradient, 0.0, sizeof(float) * reducedDim * height)); 
        cudaDeviceSynchronize();

		dim3 blocksize_g(64);
   		dim3 gridsize_g(divup(height * divup(height, blocksize_g.x) * blocksize_g.x, blocksize_g.x));
		kernal_computegradient<< <gridsize_g, blocksize_g, blocksize_g.x>> >(d_P, d_Q, d_reducedMatrix, d_distanceMatrix, height, d_gradient, reducedDim, scale, divup(height, blocksize_g.x) * blocksize_g.x);
		cudaDeviceSynchronize();
        timestamp4 = getTime();
        cout<<"compute gradient cost: "<< timestamp4 - timestamp3 << " ns\n";

		dim3 blocksize_u(512);
		dim3 gridsize_u(divup(2 * height ,blocksize_u.x));     		   
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		kernal_update<< <gridsize_u, blocksize_u>> >(d_reducedMatrix, d_m, d_g, d_gradient, reducedDim, height, lr, momentum, weightDecay);
		cudaDeviceSynchronize();
        timestamp5 = getTime();
        cout<<"update parameters cost: "<< timestamp5 - timestamp4 << " ns\n";

        if((step + 1) % 10 == 0){
            timestamp1 = getTime();
			dim3 blocksize_l(256);
			dim3 gridsize_l(divup(height * height ,blocksize_l.x));
			CHECK(cudaMemset((void *)d_loss, 0, sizeof(float))); 
            cudaDeviceSynchronize();
			kernal_compute_loss<< <gridsize_l, blocksize_l, blocksize_l.x>> >(d_P, d_Q, height, d_loss);
			cudaDeviceSynchronize();
			float * loss = new float[1];
			CHECK(cudaMemcpy((void *)loss, (void *) d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            // cudaDeviceSynchronize();
            timestamp2 = getTime();
            cout<<"compute loss cost: "<< timestamp2 - timestamp1 << " ns\n";
            std::cout << "step: " << step + 1 << " loss: " << *loss << std::endl;
        }
    }

	CHECK(cudaMemcpy((void *)reducedMatrix, (void *)d_reducedMatrix, 2 * height * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_distanceMatrix));
    CHECK(cudaFree(d_g));
	CHECK(cudaFree(d_m));
	CHECK(cudaFree(d_gradient));
	CHECK(cudaFree(d_P));
	CHECK(cudaFree(d_Q));
	CHECK(cudaFree(d_reducedMatrix));
	CHECK(cudaFree(d_loss));

    // 回收内存
    delete[] P;
    delete[] Q;
    delete[] gradient;
    delete[] distanceMatrix;
    delete[] m;
}

/**
 * 计算P，核心部分是使用二分查找找到sigma
 * @matrix: 矩阵数组
 * @width: 矩阵宽度
 * @height: 矩阵高度
 * @perplexity: 指定的困惑度
 * @P: 保存最后求出来的P
 */
void computeP(float *matrix, int width, int height, float perplexity, float* P)
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

#pragma omp parallel for num_threads(64)
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
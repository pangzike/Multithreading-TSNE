#ifndef TSNE
#define TSNE

#include<iostream>
#include<limits>
#include<random>
#include<math.h>

#include "io.h"
using namespace std;

void tSNE(float* matrix, int width, int height, float* reducedMatrix, int reducedDim);
void computeP(float *matrix, int width, int height, float perplexity, float* P);
void computeQ(float *distanceMatrix, int width, int height, float* Q);
void computeGradient(float *P, float *Q, float *Y, float *distanceMatrix, int height, float *gradient, int reducedDim);
void update(float *x, float *m, float *g, float *gradient, int width, int height, float lr, float momentum, float weightDecay);
void computeDistanceMatrix(float* matrix, int width, int height, float* distanceMatrix);
float computeEntropy(float* vector, int dim, float sigma, int k, float* e, float* s);
float computeLoss(float *P, float *Q, int height);


/**
 * @matrix: 矩阵数组，每行是一个要降维的向量
 * @width: 矩阵宽度（向量维度）
 * @height: 矩阵高度
 * @reducedMatrix: 保存降维后的矩阵
 * @reducedDim: 降维后的维度
 */
void tSNE(float *matrix, int width, int height, float *reducedMatrix, int reducedDim){
    // 超参数
    float perplexity=25, lr=1, stepNum=5, momentum=0.9, weightDecay=0.001f;
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
    float *gradient = new float[reducedDim * height];
    float *distanceMatrix = new float[height * height];
    float *m = new float[reducedDim * height];  // 梯度的指数加权平均
    float *g = new float[reducedDim * height];  // 梯度平方的指数加权平均，用于Adam算法

    for(int i = 0; i < reducedDim * height; i++){
        m[i] = 0;
        g[i] = 0;
    }

    for(int step=0; step < stepNum; step++){
        cout<< "step "<<step<<" : "<<endl;
// 计算低维距离
        timestamp1 = getTime();
        computeDistanceMatrix(reducedMatrix, reducedDim, height, distanceMatrix);
        timestamp2 = getTime();
        cout<<"compute low dimension distance cost: "<< timestamp2 - timestamp1 << " ns\n";

        computeQ(distanceMatrix, reducedDim, height, Q);
        timestamp3 = getTime();
        cout<<"compute Q cost: "<< timestamp3 - timestamp2 << " ns\n";

        computeGradient(P, Q, reducedMatrix, distanceMatrix, height, gradient, reducedDim);
        timestamp4 = getTime();
        cout<<"compute gradient cost: "<< timestamp4 - timestamp3 << " ns\n";

        update(reducedMatrix, m, g, gradient, reducedDim, height, lr, momentum, weightDecay);
        timestamp5 = getTime();
        cout<<"update parameters cost: "<< timestamp4 - timestamp3 << " ns\n";

        if((step + 1) % 10 == 0){
            timestamp1 = getTime();
            float loss = computeLoss(P, Q, height);
            std::cout << "step: " << step + 1 << " loss: " << loss << std::endl;
            timestamp2 = getTime();
            cout<<"compute loss cost: "<< timestamp2 - timestamp1 << " ns\n";
        }
    }
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
void computeP(float *matrix, int width, int height, float perplexity, float* P){
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


/**
 * 计算Q
 * @distanceMatrix: 距离矩阵
 * @width: 矩阵宽度
 * @height: 矩阵高度
 * @Q: 保存最后求出来的Q
 */
void computeQ(float *distanceMatrix, int width, int height, float* Q){
    float sum = 0.0f;
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
 */
void computeGradient(float *P, float *Q, float *Y, float *distanceMatrix, int height, float *gradient, int reducedDim){
    static int step = 0;
    step += 1;
    float scale = step > 50? 1.0f : 4.0f;

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
}


/**
 * 使用Adam算法进行更新
 */
void update(float *x, float *m, float *g, float *gradient, int width, int height, float lr, float momentum, float weightDecay){
    for(int i=0; i < width * height; i++){
        float grad = gradient[i];
        m[i] = (momentum) * m[i] + (1 - momentum) * grad;
        g[i] = (momentum) * g[i] + momentum * grad * grad;
        x[i] = (1 - weightDecay) * x[i] - lr / (sqrt(g[i]) + 1e-12) * m[i];
    }
}


/**
 * 计算loss，即P和Q之间的KL散度
 */
float computeLoss(float *P, float *Q, int height){
    float loss = 0.0f;
    for(int i=0; i < height; i++){
        for(int j=0; j < height; j++){
            float p = P[i * height + j];
            float q = Q[i * height + j];
            loss += p * log(p / q);
        }
    }
    return loss;
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

#endif

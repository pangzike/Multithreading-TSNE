//
//  main.cpp
//  TSNE
//
//  Created by 杨可 on 2022/6/11.
//

#include "tSNE.h"
#include "io.h"


int main() {
    int width=50, height=10000;
    float* matrix = read("../train60000dim50.txt", width, height);
    // printMatrix(matrix, width, height);

    int reducedDim = 2;
    float* reducedMatrix = new float[reducedDim * height];
    long start_cpu, end_cpu;
    start_cpu = getTime();
    tSNE(matrix, width, height, reducedMatrix, reducedDim);
    end_cpu = getTime();
    saveMatrix(reducedMatrix, reducedDim, height, "output.txt");
    std::cout<<"cost "<<end_cpu - start_cpu<<"ns\n";
    delete[] matrix;
    delete[] reducedMatrix;
}

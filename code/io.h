#ifndef READDATA
#define READDATA

#include<iostream>
#include<fstream>
#include<string>

#define CHECK_FILE_ERROR(f, message) {\
    if(f.fail()){\
        std::cout << message << std::endl;\
        exit(0);\
    }\
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
 * 读取矩阵，要求每行的数据之间用空格隔开
 * @path: 文件路径
 * @width: 矩阵宽度
 * @height: 矩阵高度
 */
float* read(std::string path, int width, int height) {
    std::ifstream f;
    float* data;

    f.open(path);
    CHECK_FILE_ERROR(f, "Error in read(): can not open file.");
    data = new float[width * height];
    for(int i=0; i < height; i++){
        for(int j=0; j < width; j++){
            f >> data[i * width + j];
            CHECK_FILE_ERROR(f, "Error in read(): error while reading data.");
        }
    }
    f.close();
    return data;
}


/**
 * 在终端输出矩阵
 * @matrix: 矩阵数组
 * @width: 矩阵宽度
 * @height: 矩阵高度
 */
void printMatrix(float* matrix, int width, int height){
    for(int i=0; i < height; i++){
        for(int j=0; j < width; j++){
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}


/**
 * 在文件输出矩阵
 * @matrix: 矩阵数组
 * @width: 矩阵宽度
 * @height: 矩阵高度
 */
void saveMatrix(float* matrix, int width, int height, std::string filename){
    std::ofstream f;
    f.open(filename);
    for(int i=0; i < height; i++){
        for(int j=0; j < width; j++){
            f << matrix[i * width + j] << " ";
        }
        f << std::endl;
    }
    f.close();
}

// getTime gets the local time in nanoseconds.
long getTime() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

// divup calculates n / m and would round it up if the remainder is non-zero.
int divup(int n, int m) {
    return n % m == 0 ? n/m : n/m + 1;
}
#endif

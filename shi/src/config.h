#pragma once
#include <cuda_runtime.h>

// 全局统一配置参数，由 main.cpp 定义具体数值
extern int BATCH;            // 每批处理的行数
extern int NUM_STREAMS;      // CUDA streams 数量
extern dim3 BLOCK_SIZE;      // block 维度
extern int CUDA_DEVICE_ID;   // 使用的 CUDA device 编号

// 日照计算参数
extern int DAY_OF_YEAR;      // 计算哪一天 (儒略日)
extern int TIME_STEP;        // 时间步长 (分钟)
extern float PADDING_DEGREE; // 输出结果距 DEM 边缘留白 (°) 
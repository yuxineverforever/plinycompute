#ifndef PDB_CUDA_MATRIX_MULTIPLE
#define PDB_CUDA_MATRIX_MULTIPLE

#include <iostream>
#include <cstdio>
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include <helper_cuda.h>

extern void copyFromHostToDevice(void** targetDevice, void* sourceHost, size_t bytesNum);


extern void copyFromDeviceToHost(void* targetHost, void* sourceDevice, size_t bytesNum);

extern void printCudaVersion();

extern void launchKernel(float *in1data,
                         unsigned int in1NumRow,
                         unsigned int in1NumCol,
                         float *in2data,
                         unsigned int in2NumRow,
                         unsigned int in2NumCol,
                         float *outdataGPU);

extern void initGPUMemoryToZero(void **memdata, size_t bytesNum);

extern void freeGPUMemory(void ** memdata);

#endif
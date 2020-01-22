#include "PDBCUDAUtility.h"
#define NUM_THREADS 128

void printCudaVersion() {
    std::cout << "CUDA Compiled version: " << __CUDACC_VER__ << std::endl;
    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;
    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}

void copyFromHostToDevice(void **targetDevice, void *sourceHost, size_t bytesNum) {
  checkCudaErrors(cudaMalloc((void **) targetDevice, bytesNum));
  checkCudaErrors(cudaMemcpy(*targetDevice, sourceHost, bytesNum, cudaMemcpyHostToDevice));
}

void copyFromDeviceToHost(void *targetHost, void * sourceDevice, size_t bytesNum) {
  checkCudaErrors(cudaMemcpy(targetHost, sourceDevice, bytesNum, cudaMemcpyDeviceToHost));
}

void freeGPUMemory(void ** memdata){
    checkCudaErrors(cudaFree(*memdata));
}

void initGPUMemoryToZero(void **memdata, size_t bytesNum) {
    checkCudaErrors(cudaMalloc((void **) memdata, bytesNum));
    checkCudaErrors(cudaMemset(*memdata, 0, bytesNum));
}

void launchKernel(float *in1data,
                  unsigned int in1NumRow,
                  unsigned int in1NumCol,
                  float *in2data,
                  unsigned int in2NumRow,
                  unsigned int in2NumCol,
                  float *outdataGPU) {
    cublasHandle_t handle;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    checkCudaErrors(cublasCreate(&handle));
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in2NumCol, in1NumRow, in1NumCol, &alpha, in2data, in2NumCol, in1data, in1NumCol, &beta, outdataGPU, in2NumCol));
    //matrixMulGPU <<< number_of_blocks, threads_per_block >>> (in1data, in1NumRow, in1NumCol, in2data, in2NumRow, in2NumCol, outdataGPU);
}





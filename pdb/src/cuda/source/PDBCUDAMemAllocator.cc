#include "PDBCUDAMemAllocator.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"

void* PDBCUDAMemAllocator:: MemMalloc(size_t size){
    void * temp;
    checkCudaErrors(cudaMalloc((void**)&temp, size));
    return temp;
}

void  PDBCUDAMemAllocator:: MemFree(void* freeMe){

}
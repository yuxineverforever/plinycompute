#ifndef PDB_CUDA_GLOBAL_VARS
#define PDB_CUDA_GLOBAL_VARS

#include "PDBCUDATaskManager.h"
#include "PDBCUDAVectorAddInvoker.h"
#include "PDBCUDAMatrixMultipleInvoker.h"

void* gpuMemoryManager = nullptr;
void* gpuTaskManager = nullptr;
void* gpuMemoryAllocator = nullptr;

#endif